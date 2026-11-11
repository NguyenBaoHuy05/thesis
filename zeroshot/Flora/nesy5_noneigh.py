"""Hướng 5: compose có thay được neighbor của Flora không?

Flora trộn vào embedding mỗi lớp `topk` lớp gần nhất theo CLIP (trọng số `mixed_weights`). compose chỉ CỘNG
thêm beta * primitive lên trên, neighbor vẫn chạy. Ở đây tắt neighbor (mixed_weights = --mix, mặc định 0)
và chạy Flora thường (không method riêng), để so với 2 baseline có neighbor:

  --kg-mode none     : Flora không neighbor      -> neighbor đóng góp bao nhiêu
  --kg-mode compose  : compose không neighbor    -> primitive có thay được neighbor không
                       d_best_vs_compose ~ 0 hoặc dương: thay được; âm: hai thứ bổ trợ nhau

Chạy: python nesy5_noneigh.py --kg-mode none    --feat synse --tasks 48 110 --seeds 1 2 3
      python nesy5_noneigh.py --kg-mode compose --feat synse --tasks 48 110 --seeds 1 2 3
Baseline none/compose (có neighbor) cùng seed đã chạy sẽ được bỏ qua; chưa có thì tự chạy.
"""
from nesy_common import main
from trainer import Trainer

if __name__ == '__main__':
    main(
        'noneigh', Trainer,
        method_args=[('--mix', dict(type=float, default=0.0,
                                    help='mixed_weights thay cho giá trị trong config (0 = tắt neighbor)'))],
        label_keys=[('mix', 'mix')],
        flora_from_method=lambda o: dict(mixed_weights=[o.mix, o.mix]),
        default_kg_mode='compose',
    )
