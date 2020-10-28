import time


start = time.process_time()
sum = 0
for i in range(489303872):
    sum += i
print(sum)

end = time.process_time()

print("The time is: %s " % str(end - start))