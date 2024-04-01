python3 -c "import torch; t1 = torch.tensor(list(range(10000000))).cuda(); t2 = torch.tensor(list(range(10000000))).cuda(); t1 * t2"
