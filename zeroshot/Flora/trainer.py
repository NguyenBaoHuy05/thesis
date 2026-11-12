import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from models.fm import MappingNet
import utils.utils as utils

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
        ##### ---- Logger ---- #####
        self.logger = utils.get_logger(self.args.out_dir)
        self.logger.info(f'Training Flora on {args.task_name} dataset.')
        ##### ---- Task ---- #####
        self.load_task()
        ##### ---- Load Data ---- #####
        self.load_data()
        ##### ---- Initialize MappingNet and Optimizer---- #####
        self.load_mapping_model()
        ##### ---- Initialize Seen Semantics & Skeleton Features ---- #####
        self.load_prototypes()
        ##### ---- Initialize Best Accuracy ---- #####
        self.best_accuracy = [float('-inf')]*4  # [ZSL GZSL-Seen GZSL-Unseen GZSL-H]

    
    def load_task(self):
        # load train and test data
        self.num_classes, self.unseen_classes, self.seen_classes, self.train_label_dict, self.test_zsl_label_dict, self.test_gzsl_label_dict = utils.task_definition(self.args.task_name)
        # Map dataset labels directly on the GPU.  The old code rebuilt a
        # Python list of one-hot vectors for every batch, then ran argmax.
        self.train_target_lookup = torch.full((self.num_classes,), -1, dtype=torch.long, device=self.device)
        self.train_target_lookup[self.seen_classes] = torch.arange(len(self.seen_classes), device=self.device)
        self.zsl_target_lookup = torch.full((self.num_classes,), -1, dtype=torch.long, device=self.device)
        self.zsl_target_lookup[self.unseen_classes] = torch.arange(len(self.unseen_classes), device=self.device)
        self.logger.info(f'Loading task done.')
    
    def load_data(self):
        Feeder = utils.import_class(self.args.feeder)
        self.data_loader = dict()
        self.data_loader['train'] = utils.load_data(Feeder, self.args, self.unseen_classes, utils.init_seed, self.args.train_batch_size, split_type='train', use_features=self.args.use_features, low_shot=self.args.low_shot, percentage=self.args.percentage)
        self.logger.info(f'Loading Training dataloader done.')
        self.data_loader['test_zsl'] = utils.load_data(Feeder, self.args, self.unseen_classes, utils.init_seed, self.args.test_batch_size, split_type='test_zsl', use_features=self.args.use_features, low_shot=self.args.low_shot, percentage=self.args.percentage)
        self.logger.info(f'Loading ZSL Testing dataloader done.')
        self.data_loader['test_gzsl'] = utils.load_data(Feeder, self.args, self.unseen_classes, utils.init_seed, self.args.test_batch_size, split_type='test_gzsl', use_features=self.args.use_features, low_shot=self.args.low_shot, percentage=self.args.percentage)
        self.logger.info(f'Loading GZSL Testing dataloader done.')
        self.train_loader_iter = utils.cycle(self.data_loader['train'])
        self.zsl_loader  = self.data_loader['test_zsl']
        self.gzsl_loader = self.data_loader['test_gzsl']
    
    def load_mapping_model(self):
        self.mapping_model = MappingNet(args=self.args)
        self.logger.info(f'Loading mapping network done.')
        self.mapping_model.cuda()
        self.mapping_model.train()
        self.optimizer_mapping = optim.AdamW(self.mapping_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.logger.info(f'Loading mapping network optimizer done.')
    
    def load_prototypes(self):
        self.kg_gate_text = None
        self.kg_gate_idf = None
        if self.args.setting == 'ZSL':
            self.args.mixed_weights = self.args.mixed_weights[0]
        elif self.args.setting == 'GZSL':
            self.args.mixed_weights = self.args.mixed_weights[1]
        else:
            raise NotImplementedError(f'Setting {self.args.setting} not implemented.')
        # load semantcis prototypes
        semantic_embeddings = self.mapping_model.class_embeddings  # (60, token, 768*2) 
        # load seen & unseen semantics similarity
        semantic_embeddings_mean = semantic_embeddings.mean(1)  # (60, 768*2)
        self.text_similarity = F.cosine_similarity(semantic_embeddings_mean.unsqueeze(1), semantic_embeddings_mean.unsqueeze(0), dim=-1)  # (60, 60)
        kg = None
        if self.args.kg_mode != 'none':
            from kg_semantics import PrimitiveKG
            kg = PrimitiveKG(self.args.kg_prims, self.args.kg_rules, semantic_embeddings.shape[0], self.args.kg_forbid_weight)
            if self.args.kg_mode in ('neighbors', 'both'):
                kg_sim = torch.from_numpy(kg.similarity()).to(self.text_similarity.device, self.text_similarity.dtype)
                a = self.args.kg_alpha
                self.text_similarity = (1 - a) * self.text_similarity + a * kg_sim  # self-similarity stays 1
                self.logger.info(f'KG neighbors on (alpha={a}).')
        self.topk_similarity_indices = torch.topk(self.text_similarity, k=self.args.topk+1, dim=-1).indices[:, 1:]  # (60, topk) except itself
        self.topk_similarity_values = torch.topk(self.text_similarity, k=self.args.topk+1, dim=-1).values[:, 1:]  # (60, topk)  except itself
        # KG compose: primitive-composed class semantics (neighbour selection above is unaffected)
        kg_sem = None
        if kg is not None and self.args.kg_mode in ('compose', 'both'):
            import os
            from kg_semantics import embed_texts, compose_semantics, compose_text_semantics, mask_half
            m = self.mapping_model
            kg_dir = os.path.dirname(os.path.abspath(self.args.kg_prims))
            p_stem = os.path.splitext(os.path.basename(self.args.kg_prims))[0]
            r_stem = os.path.splitext(os.path.basename(self.args.kg_rules))[0]
            width = semantic_embeddings.shape[-1]
            if self.args.kg_learned_gate:
                if self.args.kg_compose_at != 'post':
                    raise ValueError('kg_learned_gate currently supports kg_compose_at=post only')
                prim_cache = os.path.join(kg_dir, f'{p_stem}_clip_L{self.args.max_length}.pt')
                text_cache = os.path.join(kg_dir, f'{r_stem}_classtext_n{semantic_embeddings.shape[0]}_L{self.args.max_length}.pt')
                prim_emb = embed_texts(m.tokenizer, m.text_encoder, kg.prim_text, self.args.max_length, 'cuda', prim_cache).to(semantic_embeddings.device)
                text_sem = compose_text_semantics(kg, m.tokenizer, m.text_encoder, self.args.max_length, width, 'cuda', text_cache)
                idf_sem = compose_semantics(kg, prim_emb, width, weighting='idf')
                self.kg_gate_text = mask_half(text_sem.to(semantic_embeddings.device, semantic_embeddings.dtype), self.args.kg_compose_half)
                self.kg_gate_idf = mask_half(idf_sem.to(semantic_embeddings.device, semantic_embeddings.dtype), self.args.kg_compose_half)
                self.logger.info(f'KG text-compose + IDF + learned gate on (beta={self.args.kg_beta}, '
                                 f'half={self.args.kg_compose_half}).')
            elif self.args.kg_compose_src == 'mix':
                # same two branches as the learned gate, with a constant 0.5 instead of a trained sigmoid
                prim_cache = os.path.join(kg_dir, f'{p_stem}_clip_L{self.args.max_length}.pt')
                text_cache = os.path.join(kg_dir, f'{r_stem}_classtext_n{semantic_embeddings.shape[0]}_L{self.args.max_length}.pt')
                prim_emb = embed_texts(m.tokenizer, m.text_encoder, kg.prim_text, self.args.max_length, 'cuda', prim_cache).to(semantic_embeddings.device)
                text_sem = compose_text_semantics(kg, m.tokenizer, m.text_encoder, self.args.max_length, width, 'cuda', text_cache)
                idf_sem = compose_semantics(kg, prim_emb, width, weighting='idf')
                kg_sem = 0.5 * text_sem.to(idf_sem.device, idf_sem.dtype) + 0.5 * idf_sem
            elif self.args.kg_compose_src == 'avg':
                cache = os.path.join(kg_dir, f'{p_stem}_clip_L{self.args.max_length}.pt')
                prim_emb = embed_texts(m.tokenizer, m.text_encoder, kg.prim_text, self.args.max_length, 'cuda', cache).to(semantic_embeddings.device)
                weighting = 'idf' if self.args.kg_compose_idf else 'uniform'
                kg_sem = compose_semantics(kg, prim_emb, width, weighting=weighting)
            else:
                cache = os.path.join(kg_dir, f'{r_stem}_classtext_n{semantic_embeddings.shape[0]}_L{self.args.max_length}.pt')
                kg_sem = compose_text_semantics(kg, m.tokenizer, m.text_encoder, self.args.max_length, width, 'cuda', cache)
            if kg_sem is not None:
                kg_sem = mask_half(kg_sem.to(semantic_embeddings.device, semantic_embeddings.dtype), self.args.kg_compose_half)
                self.logger.info(f'KG compose on (beta={self.args.kg_beta}, src={self.args.kg_compose_src}, '
                                 f'at={self.args.kg_compose_at}, half={self.args.kg_compose_half}).')
        # 'pre': every class carries its composed semantics into the neighbour mixing
        base = semantic_embeddings
        if kg_sem is not None and self.args.kg_compose_at == 'pre':
            base = semantic_embeddings + self.args.kg_beta * kg_sem
        # computing neighbor semantics
        self.attuned_semantics = torch.zeros(semantic_embeddings.size(), device=semantic_embeddings.device)  # seen+unseen token 768*2
        for k in range(self.args.topk):
            topi_semantics = torch.zeros(self.attuned_semantics.size(),device=semantic_embeddings.device)  # seen+unseen token 768*2
            for i in range(self.attuned_semantics.shape[0]):
                topi_semantics[i] = (self.topk_similarity_values[i, k] * base[self.topk_similarity_indices[i, k], :, :]).unsqueeze(0)
            self.attuned_semantics += self.args.mixed_weights / self.args.topk * topi_semantics  # seen+unseen token 768*2
        self.attuned_semantics += base # seen+unseen token 768*2
        if kg_sem is not None and self.args.kg_compose_at == 'post':
            self.attuned_semantics += self.args.kg_beta * kg_sem
        if self.kg_gate_text is not None:
            self.log_gate_stats('Initial')
        self.logger.info(f'Attuned Semantics Successfully.')

    def current_attuned_semantics(self, return_gate=False):
        """Return static Flora semantics, optionally fused with the trainable KG gate."""
        if self.kg_gate_text is None:
            return (self.attuned_semantics, None) if return_gate else self.attuned_semantics
        semantics, gate = self.mapping_model.compose_with_learned_gate(
            self.attuned_semantics, self.kg_gate_text, self.kg_gate_idf, self.args.kg_beta)
        return (semantics, gate) if return_gate else semantics

    def log_gate_stats(self, stage):
        if self.kg_gate_text is None:
            return
        with torch.no_grad():
            _, gate = self.current_attuned_semantics(return_gate=True)
        self.logger.info(f'{stage} text-compose gate: mean={gate.mean().item():.3f}, '
                         f'min={gate.min().item():.3f}, max={gate.max().item():.3f}.')
    
    def train_learning_phase(self):
        ###### ---- Phase 1: Neighbor-aware Semantic Learning ---- #####
        avg_loss, avg_mu, avg_logvar, avg_recon = 0., 0., 0., 0.
        for nb_iter in range(1, self.args.align_iter+ 1):
            gt_motion, label_idx, _, _,  _ = next(self.train_loader_iter)
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True) # (bs, 2, 16, 512)
            label_idx = label_idx.to(self.device, non_blocking=True) # (bs, )
            targets = self.train_target_lookup[label_idx]
            # train
            metrics = self.mapping_model.forward_phase1(gt_motion, self.current_attuned_semantics(), self.seen_classes, targets)  # (bs, 2, 16, 512)
            # Combined loss
            loss = metrics['loss']
            loss_mu = metrics['loss_mu']
            loss_logvar = metrics['loss_log_var']
            loss_recon = metrics['loss_recon']
            self.optimizer_mapping.zero_grad()
            loss.backward()
            self.optimizer_mapping.step()
            avg_loss += loss.item()
            avg_mu += loss_mu.item()
            avg_logvar += loss_logvar.item()
            avg_recon += loss_recon.item()
            # print
            if nb_iter % self.args.print_iter ==  0 :
                avg_loss /= self.args.print_iter
                avg_mu /= self.args.print_iter
                avg_logvar /= self.args.print_iter
                avg_recon /= self.args.print_iter
                self.logger.info(f"Train Phase 1. Iter {nb_iter} : Loss. {avg_loss:.5f} \t Loss_mu. {avg_mu:.5f} \t Loss_logvar. {avg_logvar:.5f} \t Loss_Recon. {avg_recon:.5f}")
                avg_loss, avg_mu, avg_logvar, avg_recon = 0., 0., 0., 0.
        self.logger.info(f'Phase 1: Neighbor-aware Semantic Learning Done.')
        self.log_gate_stats('After Phase 1')


    def train_deciding_phase(self):
        ###### ---- Phase 2: Open-form Flow Deciding ---- #####
        avg_loss, avg_fm, avg_fm_con = 0., 0., 0.
        for nb_iter in range(self.args.align_iter+ 1, self.args.total_iter+ 1):
            gt_motion, label_idx, gt_motion_contrastive, label_idx_contrastive, _ = next(self.train_loader_iter)
            gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True) # (bs, 2, 16, 512)
            gt_motion_contrastive = gt_motion_contrastive.to(self.device, dtype=torch.float32, non_blocking=True) # (bs, 2, 16, 512)
            label_idx = label_idx.to(self.device, non_blocking=True) # (bs, )
            label_idx_contrastive = label_idx_contrastive.to(self.device, non_blocking=True) # (bs, )
            targets = self.train_target_lookup[label_idx]
            contrastive_targets = self.train_target_lookup[label_idx_contrastive]
            # train
            metrics = self.mapping_model.forward_phase2(gt_motion, gt_motion_contrastive, targets, contrastive_targets, self.seen_classes, self.current_attuned_semantics())
            # Combined loss
            loss = metrics['loss'] 
            loss_fm = metrics['loss_flow_matching']
            loss_fm_con = metrics['loss_flow_matching_contrastive']
            self.optimizer_mapping.zero_grad()
            loss.backward()
            self.optimizer_mapping.step()
            avg_loss += loss.item()
            avg_fm += loss_fm.item()
            avg_fm_con += loss_fm_con.item()
            # print
            if nb_iter % self.args.print_iter ==  0 :
                avg_loss /= self.args.print_iter
                avg_fm /= self.args.print_iter
                avg_fm_con /= self.args.print_iter
                self.logger.info(f"Train Phase 3. Iter {nb_iter} : Loss. {avg_loss:.5f} \t Loss FM. {avg_fm:.5f} \t Loss FM Con. {avg_fm_con:.5f}")
                avg_loss, avg_fm, avg_fm_con = 0., 0., 0.

            # test
            if nb_iter % self.args.eval_iter==0:
                if self.args.setting == 'ZSL':
                    self.test_zsl(nb_iter)
                elif self.args.setting == 'GZSL':
                    self.test_gzsl(nb_iter)
                elif self.args.setting == 'All':
                    self.test_zsl(nb_iter)
                    self.test_gzsl(nb_iter)
                else:
                    raise NotImplementedError(f'Setting {self.args.setting} not implemented.')
        
        # print best accuracy
        self.print_best_accuracy()


    def test_zsl(self, nb_iter):
        self.mapping_model.eval()
        total_samples = 0
        correct_predictions = 0
        for batch_idx, (gt_motion, label_idx, _, _, _) in enumerate(self.zsl_loader):
            with torch.no_grad():
                gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True) # (bs, 2, 16, 512)
                label_idx = label_idx.to(self.device, non_blocking=True)
                labels = self.zsl_target_lookup[label_idx]
                # calculate accuracy -- velocity error
                res_dis = self.mapping_model.predict_zsl(gt_motion, self.unseen_classes, self.current_attuned_semantics(), labels)
                predictions = res_dis['predictions_idx']  # (bs,)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += gt_motion.size(0)
        accuracy_zsl = 100 * correct_predictions / total_samples
        # update best accuracy
        self.best_accuracy[0] = max(self.best_accuracy[0], accuracy_zsl)
        self.logger.info(f"Iter {nb_iter}: ZSL Accuracy: {accuracy_zsl:.2f}% ")
        self.mapping_model.train()
    
    def test_gzsl(self, nb_iter):
        self.mapping_model.eval()
        total_samples = 0
        correct_predictions = 0
        label_statistics = []
        pred_list = []
        for batch_idx, (gt_motion, label_idx, _, _, _) in enumerate(self.gzsl_loader):
            with torch.no_grad():
                gt_motion = gt_motion.to(self.device, dtype=torch.float32, non_blocking=True) # (bs, 2, 16, 512)
                label_idx = label_idx.to(self.device, non_blocking=True)
                labels = label_idx
                # calculate accuracy -- velocity error
                res_dis = self.mapping_model.predict_gzsl(gt_motion, self.seen_classes, self.unseen_classes, self.current_attuned_semantics())
                predictions = res_dis['predictions_idx']  # (bs, num_noise)
                correct_predictions += (predictions == labels).sum().item()
                label_statistics.append(labels.cpu().numpy())  # 记录标签统计
                pred_list.append(predictions.cpu().numpy())  # (bs, n)
                total_samples += gt_motion.size(0)

        # store statistics 
        label_statistics = np.concatenate(label_statistics, axis=0)  # n
        pred_list = np.concatenate(pred_list, axis=0)  # n
        # gzsl
        seen_total_samples, unseen_total_samples = 0, 0
        seen_correct_samples, unseen_correct_samples = 0, 0
        for ture_label, pred_label in zip(label_statistics, pred_list):
            if ture_label in self.seen_classes:
                seen_total_samples += 1
                if ture_label == pred_label:
                    seen_correct_samples += 1
            else:
                unseen_total_samples += 1
                if ture_label == pred_label:
                    unseen_correct_samples += 1
        seen_accuracy = 100 * seen_correct_samples / seen_total_samples if seen_total_samples > 0 else 0
        unseen_accuracy = 100 * unseen_correct_samples / unseen_total_samples if unseen_total_samples > 0 else 0
        if (seen_accuracy + unseen_accuracy) > 0:
            harmonic_mean = 2 * (seen_accuracy * unseen_accuracy) / (seen_accuracy + unseen_accuracy)
        else:
            harmonic_mean = 0
        self.logger.info(f"Iter {nb_iter}: Seen Accuracy: {seen_accuracy:.2f}% \t Unseen Accuracy: {unseen_accuracy:.2f}% \t Harmonic Mean: {harmonic_mean:.2f}%")
        # store best model
        if harmonic_mean > self.best_accuracy[3]:
            # update best accuracy
            self.best_accuracy[1] = seen_accuracy
            self.best_accuracy[2] = unseen_accuracy
            self.best_accuracy[3] = harmonic_mean
        self.mapping_model.train()


    def print_best_accuracy(self):
        if self.args.setting == 'ZSL':
            self.logger.info(f'Best ZSL Results: {self.best_accuracy[0]:.2f}%')
        elif self.args.setting == 'GZSL':
            self.logger.info(f'Best GZSL Results: Seen: {self.best_accuracy[1]:.2f}% Unseen: {self.best_accuracy[2]:.2f}% H: {self.best_accuracy[3]:.2f}%')
        else:
            raise NotImplementedError(f'Setting {self.args.setting} not implemented.')
    
    def train(self):
        self.train_learning_phase()
        self.train_deciding_phase()
