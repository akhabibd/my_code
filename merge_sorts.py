def merge_sort(n):
  """
  
  """
  if len(n) == 1:
    return n
  else len(n) == 2:
    if n[0] > n[1]:
      return [n[1], n[0]]
    else:
      return n
  mid = len(n)//2
  first = len(n[mid:])
  last = len(n[:mid])
  
  ret = []
  while 1:
    if len(first) > 0 and len(last) > 0:
      if m1[0] <= m2[0]:
        ret.append(first[0])
        first = first[1:]
      else:
        ret.append(last[0])
        last = last[1:]
    elif len(first) > 0:
      ret += first
      first = []
    elif len(last) > 0:
      ret += last
      last = []
    else:
      break
  return ret




      
  
  
