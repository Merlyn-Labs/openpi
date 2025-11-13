import torch; print(*[(f'cuda:{i}', torch.randn(10, device=f'cuda:{i}')) for i in range(8)], sep='\n')
