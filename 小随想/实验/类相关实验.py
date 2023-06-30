# 类对象创建可选择定义类属性，创建实例对象时，实例属性自动执行类的__init__方法初始化
# 实例对象自动继承相应的类属性（如果有），但实例属性优先级更高
# 实例方法是一般函数但实例方法需要传入self参数（与一般函数的区别）
# 类方法和静态方法是通过装饰器实现的函数，类方法需要传入cls参数，静态方法无需传入self参数或者是cls参数（但也能传入参数）
# 其中self参数必须指向实例对象，cls参数指向定义的类对象（self与cls是约定俗成的，原则上可以用其它名字）
# 其中方法是在对象语境下的函数，实例调用实例方法即调用方法，类调用实例方法即调用函数
# super(C, self).init()等价super().init()


# %%
class Toy(object):
    count = 0
    
    def who_toy(*args):
        print(Toy)
        
    @classmethod
    def show_toy_count(cls):
        print('玩具对象的数量 %d' % cls.count, cls)

    @staticmethod
    def hi():
        print('Hello!')

    def __init__(self, name):
        self.name = name
        Toy.count += 1

    def beybey(self):
        self.hi()
        print('Sad！', self)


# 类对象
print(Toy.count)
Toy.who_toy()
Toy.show_toy_count()
Toy.hi()
# Toy.beybey()  # 错误语法，self必须指向实例对象，此处实例方法指向类对象而不是实例对象

# 实例对象
toy1 = Toy('泰迪熊')
toy1.hand = 2
toy1.who_toy()
print(toy1.name)
print(toy1.hand)
# 实例对象调用类方法时，与类对象调用类方法无异，但实际上调用仍通过实例对象继承的类对象
toy1.show_toy_count()
# 实例对象调用静态方法时，与类对象调用静态方法无异，但实际上调用仍通过实例对象继承的类对象
toy1.hi()
# 实例对象调用实例方法，Python的解释器内部，当我们调用toy1.beybey()时，实际上Python解释成Toy.beybey(toy1)
toy1.beybey()
Toy.beybey(toy1)
# 隐式实例化掉用实例方法
Toy('哥斯拉').beybey() 
# print(Toy.hand)  # AttributeError: type object 'Toy' has no attribute 'hand'

# 类与其实例的类型和内存地址
print(type(Toy), id(Toy), type(toy1), id(toy1))

# %%
# 类方法与静态方法辨析
class Cat:  # 或者class Cat()不写父对象形式定义类对象，会默认继承祖先object对象
    name = '小敏'
        
    def smile():
        print('哈哈～')
      
    def sad():
        print('呜呜～')
        Cat.run()
        Cat.set()

    @classmethod
    def run(cls):
        print('起飞～')
    
    @classmethod   
    def say(cls):
        print('%s，你好！' % cls.name)
        cls.set()  # 类方法可以调用静态方法
        
    @staticmethod
    def stand():
        print('{}，站着！'.format(Cat.name))
        Cat.run()  # 静态方法可以调用类方法
    
    @staticmethod
    def set():
        print('{}，坐下！'.format(Cat.name))
                   
    def __init__(self):
        print(self.name)
        self.name = '胖虎'

    def sleep(self):
        print('呼呼～') 
   

print(Cat.name)
Cat.smile()
Cat.sad()
Cat.run()
Cat.say()
Cat.stand()
Cat.set()
# Cat.sleep() TypeError: sleep() missing 1 required positional argument: 'self'

cat1 = Cat()
print(cat1.name)
# cat1.smile() TypeError: smile() takes 0 positional arguments but 1 was given
# cat1.sad() TypeError: sad() takes 0 positional arguments but 1 was given
cat1.run()
cat1.say()
cat1.stand()
cat1.set()
cat1.sleep()

# %%

class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f'Vector({self.x}, {self.y})'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


v1 = Vector(3, 4)
v2 = Vector(5, 6)
print(Vector(3, 4) + Vector(3, 4))
print(v1 == v2)

# %%
