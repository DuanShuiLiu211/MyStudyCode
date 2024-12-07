import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional


# 1. 基础使用
@dataclass
class Point:
    x: float
    y: float

    def distance_from_origin(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5


# 2. 默认值
@dataclass
class Rectangle:
    width: float
    height: float = 10.0  # 默认值
    color: str = field(default="white")  # 使用field定义默认值
    timestamp: datetime = field(default_factory=datetime.now)  # 动态默认值

    def area(self) -> float:
        return self.width * self.height


# 3. 不可变数据类
@dataclass(frozen=True)
class Config:
    host: str
    port: int
    debug: bool = False

    def get_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# 4. 继承
@dataclass
class Animal:
    name: str
    breed: str
    species: str = "Unknown"
    age: int = 0


@dataclass
class Dog(Animal):
    color: str = "Unknown"


# 5. 高级字段选项
@dataclass
class User:
    username: str
    password: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    friends: List[str] = field(default_factory=list)
    show_password: bool = field(default=False, repr=False)

    def __post_init__(self):
        if not self.show_password:
            self.password = "******"


# 6. 初始化后处理
@dataclass
class Temperature:
    celsius: float
    unit: str
    fahrenheit: float = field(init=False)

    def __post_init__(self):
        """初始化后处理"""
        if self.unit == "F":
            self.celsius = (self.celsius - 32) * 5 / 9
        self.fahrenheit = (self.celsius * 9 / 5) + 32


# 7. 验证和类型检查
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    tax_rate: ClassVar[float] = 0.1  # 类变量

    def __post_init__(self):
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")

    @property
    def total_price(self) -> float:
        return self.price * self.quantity * (1 + self.tax_rate)


# 8. 自定义序列化
@dataclass
class Employee:
    id: int
    name: str
    department: str
    salary: float = field(repr=False)  # 不在repr中显示薪资

    def to_dict(self) -> Dict[str, Any]:
        """自定义字典转换"""
        return {"id": self.id, "name": self.name, "department": self.department}

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Employee":
        """从字典创建实例"""
        return cls(**data)


# 9. 复杂嵌套数据类
@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str


@dataclass
class Person:
    name: str
    age: int
    address: Address
    contacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_address(self, **kwargs):
        """更新地址信息"""
        self.address = replace(self.address, **kwargs)


# 10. 实际应用示例
@dataclass
class APIResponse:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def is_error(self) -> bool:
        return not self.success and self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


# 使用示例
def demo_dataclass_usage():
    # 基础使用
    point = Point(3.0, 4.0)
    print(f"Point: {point}")

    # 默认值
    rect = Rectangle(width=5.0)
    print(f"Rectangle area: {rect.area()}")

    # 不可变数据类
    config = Config("localhost", 8080)
    print(f"URL: {config.get_url()}")

    # 继承
    dog = Dog(name="Rex", breed="Labrador")
    print(f"Dog: {dog}")

    # 高级字段选项
    user = User("john_doe", "password123", "john@example.com")
    print(f"User: {user}")  # password不会显示

    # 初始化后处理
    temp = Temperature(32, "F")
    print(f"Temperature: {temp.celsius}°C, {temp.fahrenheit}°F")

    # 验证和类型检查
    try:
        product = Product("Phone", -100)  # 会抛出异常
    except ValueError as e:
        print(f"Validation error: {e}")

    # 序列化
    employee = Employee(1, "Alice", "IT", 50000)
    print(f"Employee JSON: {employee.to_json()}")

    # 嵌套数据类
    address = Address("123 Main St", "Boston", "USA", "02108")
    person = Person("Bob", 30, address)
    person.update_address(city="New York")
    print(f"Person: {person}")

    # API响应示例
    response = APIResponse(True, data={"message": "Success"})
    print(f"API Response: {response.to_dict()}")


if __name__ == "__main__":
    demo_dataclass_usage()
