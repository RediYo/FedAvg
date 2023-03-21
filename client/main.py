from pathlib2 import Path

p = Path(r"../dataset")
print([path for path in p.rglob("*_*[!_][!t][!a][!g].csv")])
