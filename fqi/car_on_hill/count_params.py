import sys
sys.path.insert(0, '../..')

from fqi.network_fqi import Q_Network

net = Q_Network()
print(net)
print()
for name, p in net.named_parameters():
    print(f"  {name:30s}  {list(p.shape)}  →  {p.numel()}")
print()
print(f"Total: {sum(p.numel() for p in net.parameters())}")
