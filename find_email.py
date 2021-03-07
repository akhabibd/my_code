import re
#regex for email
email = re.compile(r'([A-Za-z0-9!#$%&\'*+-/_?^{}|~\.]+@(?:[A-Za-z0-9-]+\.)+([A-Za-z0-9-])+)') 

list = []
for i in range(int(input())):
  for x in email.findall(input()):
    list.append(x)
    
print(';'.joint(sorted(list(set(list)))))
