# 类对象创建可选择定义类属性，创建实例对象时，实例属性自动执行类的__init__方法初始化
# 实例对象自动继承相应的类属性（如果有），但实例属性优先级更高
# 实例方法是一般函数但实例方法需要传入self参数（与一般函数的区别）
# 类方法和静态方法是通过装饰器实现的函数，类方法需要传入cls参数，静态方法无需传入self参数或者是cls参数（但也能传入参数）
# 其中self参数必须指向实例对象，cls参数指向定义的类对象（self与cls是约定俗成的，原则上可以用其它名字）
# 其中方法是在对象语境下的函数，实例调用实例方法即调用方法，类调用实例方法即调用函数
# super(C, self).init()等价super().init()

# 类举例
class Toy(object):  # 此处此类可理解为设计一个Toy的蓝图
    # 赋值定义类属性，记录所有玩具数量
    count = 2
    print(count)

    def __init__(self, name):  # 用于实例初始化
        self.name = name
        # 类属性 +1
        Toy.count += 1
        print(self.count)

    @classmethod  # 此装饰器表示是类方法，类方法无需创建实例对象即可调用，最为灵活
    def show_toy_count(cls):
        print('玩具对象的数量 %d' % cls.count, cls)

    @staticmethod  # 此装饰器表示是静态方法，静态方法本质上是封装在类对象内的的函数，常用于测试
    def hi():
        print('Hello!')

    # 实例方法
    def beybey(self):
        print('Sad！', self)


class Fll(Toy):
    def gugugug(self):
        pass


# 创建实例对象
toy1 = Toy('乐高')
toy1.hand = 2
toy2 = Toy('泰迪熊')
toy3 = Toy('哥斯拉')
print(toy1.name, toy2.name, toy3.name)

# 点语法调用类方法与静态方法，如：类名.方法
Toy.show_toy_count()
Toy.hi()
# 实例对象调用类方法时，与类对象调用类方法无异，但实际上调用仍通过实例对象继承的类对象
toy1.show_toy_count()
print(toy1.hand)
# 实例对象调用静态方法时，与类对象调用静态方法无异，但实际上调用仍通过实例对象继承的类对象
toy2.hi()
# 实例对象调用实例方法,Python的解释器内部，当我们调用toy3.beybey()时，实际上Python解释成Toy.beybey(toy3)
toy3.beybey()
# Toy.beybey()  # 错误语法，self必须指向实例对象，此处实例方法指向类对象而不是实例对象
Toy.beybey(toy3)
# 类与其实例的类型和内存地址
print(type(Toy), id(Toy), type(toy3), id(toy3))
# 子类继承父类的全部属性与方法
fll1 = Fll('芭比娃娃')
fll1.hi()
fll1.show_toy_count()
fll1.beybey()


# 类方法与静态方法辨析
class Cat:  # 或者class Cat()不写父对象形式定义类对象，会默认继承祖先object对象
    name = '小敏'

    def __init__(self, weight):
        self.weight = weight
        print(self.weight)

    @classmethod
    def www(cls):
        print('%s 干嘛！' % cls.name)
        # cls.call()  # 类方法可以调用静态方法

    @staticmethod
    def call():
        print('喵喵～')
        Cat.name = '小敏臭弟弟'
        print(Cat.name)
        # Cat.www()  # 静态方法可以调用类方法


class Dog(Cat):
    # def __init__(self, name):
    #     Cat.__init__(self, 120)
    #     self.name = '啊三'
    #     print(self.name)
    def __init__(self, age, weight):
        super().__init__(weight)
        self.age = age

    def yyy(self):
        print('打%s' % Dog.name)



Cat.www()
Cat.call()
# 没有定义实例方法可以创建实例对象继承并使用其中方法
cat1 = Cat(120)
cat1.www()
cat1.call()
dog1 = Dog(6, 120)
dog1.yyy()
print(dog1.name, dog1.age)
# 祖先对象中包含的基本方法
print(dir(object))


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x , self.y + other.y)

    def __repr__(self):
        return f'Vector({self.x}, {self.y})'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


v1 = Vector(3, 4)
v2 = Vector(5, 6)
print(Vector(3, 4) + Vector(3, 4))
print(v1 == v2)