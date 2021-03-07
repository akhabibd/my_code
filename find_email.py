import re

email = re.compile(r'([A-Za-z0-9!#$%&*+-/=?^_`{|}~\.]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z0-9-]+)') 

output = []
for i in range(int(input())):
  for x in email.findall(input()):
    output.append(x)
print(';'.join(sorted(list(set(output)))))
