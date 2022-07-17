import multiprocessing
def function():
    global fun1
    a="string"
    def fun1():
        print(a)
    var1=multiprocessing.Process(target=fun1)
    if __name__ == "__main__":
        var1.start()
        var1.join()
function()
print("Program finished")