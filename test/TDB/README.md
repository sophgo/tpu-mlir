
## generate new test and replace

NOTE: Please verify and guarantee the consistency of major information (cmd type, address, attribute...) between the new and old version before replacing test files!

NOTE: Please verify and guarantee the consistency of major information (cmd type, address, attribute...) between the new and old version before replacing test files!

```
cd bmodel
ls *.bmodel | xargs -I {} sh -c "bmodel_dis.py {} | sed 's/^/CHECK:  /' > {}.mlir.test" \;
ls *.bmodel | xargs -I {} sh -c "bmodel_dis.py --format reg-set {} | sed 's/^/CHECK:  /' > {}.reg.test" \;

ls *.bmodel | xargs -I {} sh -c "sed -i '1s/^/RUN: bmodel_dis.py %p\/bmodel\/{} | FileCheck %s\n\n/' {}.mlir.test" \;
ls *.bmodel | xargs -I {} sh -c "sed -i '1s/^/RUN: bmodel_dis.py --format reg-set %p\/bmodel\/{} | FileCheck %s\n\n/' {}.reg.test" \;

mv -f *.mlir.test ../
mv *.reg.test ../
cd ../
```

then run followed script to pick lines to be checked:
```python
import os
import numpy as np

for f in os.listdir("./"):
    if f.endswith("mlir.test"):
        with open(f) as r:
            lines = r.readlines()
        new_lines = lines[:3]
        for index in np.linspace(4, len(lines) - 10, 10).tolist():
            index = int(index)
            new_lines.append(lines[index])
            for i in range(index + 1, index + 6):
                new_lines.append(lines[i].replace("CHECK:", "CHECK-NEXT:"))
        with open(f, "w") as w:
            w.write("".join(new_lines))
    elif f.endswith("reg.test"):
        with open(f) as r:
            lines = r.readlines()
        new_lines = lines[:3]
        new_lines.extend([i.replace("CHECK:", "CHECK-NEXT:") for i in lines[3:10]])
        for index in np.linspace(11, len(lines) - 1, 50).tolist():
            index = int(index)
            new_lines.append(lines[index])
        with open(f, "w") as w:
            w.write("".join(new_lines))
```
