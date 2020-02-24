from utils import *

'''tests start '''


def some_test():
    return True


def other_test():
    return False


'''tests end '''


def res_string(res):
    c = Colors()
    fail = [CT.bold, CT.red]
    success = [CT.green]
    if res is True:
        return c.cs(success, "Success")
    return c.cs(fail, "Fail")


def test_runner():
    print("some_test {}".format(res_string(some_test())))
    print("other_test {}".format(res_string(other_test())))


def main():
    test_runner()


if __name__ == '__main__':
    main()