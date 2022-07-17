import multiprocessing as mp
import random
import time
from typing import TypeVar, Callable, Tuple, Iterable, List, Union
import traceback

T = TypeVar("T")
ArgList = TypeVar("ArgList")


class MPException:
    def __init__(self, e, trace):
        self.e = e
        self.trace = trace


class SafeFunc:
    def __init__(self, fu: Callable[[ArgList], T]):
        self.f = fu

    def __call__(self, *args, **kwargs) -> Union[T, MPException]:
        try:
            result = self.f(*args, **kwargs)
        except Exception as e:
            return MPException(e, traceback.format_exc())
        return result


def better_starmap(func: Callable[[ArgList], T],
                   iterable: Iterable[ArgList],
                   chunksize=None) -> Tuple[List[T], List[MPException]]:

    safe_func = SafeFunc(func)
    correct_results = []
    exceptions = []

    with mp.Pool() as pool:
        results = pool.starmap(safe_func, iterable, chunksize)
        for r in results:
            if isinstance(r, MPException):
                exceptions.append(r)
            else:
                correct_results.append(r)

    return correct_results, exceptions


def f(args):
    x,y,z = args
    time.sleep(x)
    if x + y + z == 9:
        raise IndexError("Wrong combination!")
    return y + z


def main():
    arguments = [(3, 2, 3), (2, 3, 4), (0.3, 4, 5), (0.4, 5, 6), (0.1, 6, 7), (0.06, 7, 8), (0, 8, 9)]

    results, exceptions = better_starmap(f, arguments)
    print(results)
    print("Exceptions:")
    for e in exceptions:
        print(e.trace)


def g(x):
    s = f"{x} -> {x**1.5:.2f}"
    if x % 11 == 0:
        time.sleep(1)
        s += " with sleep"
    if random.random() < 0.1:
        raise AttributeError("oh shiiet!")
    return s


def main2():
    arguments = range(100)
    safe_func = SafeFunc(g)
    with mp.Pool(4) as pool:

        it = pool.imap_unordered(safe_func, arguments, chunksize=4)
        for r in it:
            print(r)


if __name__ == "__main__":
    main2()
