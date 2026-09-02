from pathlib import Path

src_file = Path(r"d:\Thesis\ECR-SkateFormer\model\SkateFormer.py")
dst_file = Path(r"d:\Thesis\SFB-ECR-SkateFormer\model\SkateFormer.py")

with open(src_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace forward_features_from_input to also include intermediate methods
target = "    def forward_features_from_input(self, input, index_t=None):"

replacement = """    def _stem_and_embeddings(self, input, index_t=None):
        B, C, T, V, M = input.shape

        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]
        for layer in self.stem:
            output = layer(output)
        if self.index_t:
            if index_t is None:
                index_t = torch.linspace(-1., 1., T, device=output.device, dtype=output.dtype)
                index_t = index_t.unsqueeze(0).expand(B, T)
            te = torch.zeros(B, T, self.embed_dim, device=output.device, dtype=output.dtype)  # B, T, C
            div_term = torch.exp(
                (torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=output.device) * -(math.log(10000.0) / self.embed_dim))).to(
                output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum('b t c, c v -> b c t v', te, self.joint_person_embedding)
        else:
            output = output + self.joint_person_temporal_embedding
        return output

    def forward_features_early(self, input, early_stages=2, index_t=None):
        output = self._stem_and_embeddings(input, index_t=index_t)
        for stage in self.stages[:early_stages]:
            output = stage(output)
        return output

    def forward_features_late(self, modulated_feature, from_stage=2):
        output = modulated_feature
        for stage in self.stages[from_stage:]:
            output = stage(output)
        return output

    def forward_features_with_intermediate(self, input, feedback_fn=None, early_stages=2, index_t=None):
        F_early = self.forward_features_early(input, early_stages=early_stages, index_t=index_t)
        if feedback_fn is not None:
            F_mod, fb_aux = feedback_fn(F_early)
        else:
            F_mod = F_early
            fb_aux = {}
        F_late = self.forward_features_late(F_mod, from_stage=early_stages)
        return F_early, F_late, fb_aux

    def forward_features_from_input(self, input, index_t=None):"""

new_content = content.replace(target, replacement)
with open(dst_file, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Updated {dst_file} successfully.")
