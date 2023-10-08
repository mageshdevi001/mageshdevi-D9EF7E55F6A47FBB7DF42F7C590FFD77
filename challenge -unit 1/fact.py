
def fun(n):
  if n==0 or n==1:
    return 1
  else:
    return n*fun(n-1)
num=5
res=fun(num)
print("the fact is:",res)