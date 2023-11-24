class A:
    def test(self):
        print("A.test")


class TestMixin:
    def test(self):
        print("TestMixin.test")
        super(TestMixin, self).test()


class B(TestMixin, A):
    def test(self):
        print("B.test")
        super(B, self).test()


print(B.__mro__)
b = B()
b.test()
