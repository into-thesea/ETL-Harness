# ToolBroker 核心概念学习问答

> 本文档整理了学习 `harness/tool_broker.py` 过程中提出的关键疑问、解答以及理解纠正。
> 每个疑问标注了理解状态：✅ 正确 / ⚠️ 理解不全 / ❌ 理解错误已纠正。
> 最后附大模型调用工具的完整流程图。

---

## 一、类型标注

### 1.1 标注是什么？标注的是传入函数的参数吗？

**疑问**："标注什么的类型，传入函数的参数的吗"

**解答**：类型标注可以标注三种东西：
- **函数参数**：`def register(self, tool_def: ToolDef, handler: ToolHandler)`
- **返回值**：`-> None`、`-> tuple[bool, str, dict]`
- **变量**：`self._tools: dict[str, ...] = {}`

冒号后面写的就是"这个东西应该是什么类型"。

**关键**：标注**运行时不强制检查**——你真把字符串传给 `tool_def`，Python 也不报错。标注是给人看和给 IDE 提示用的。

**理解状态**：⚠️ 理解不全（以为只标注参数，实际还标注返回值和变量）

---

### 1.2 `dict[str, ...]` 后面的方括号是标注键值类型吗？

**疑问**："所以说，这个字典后面用[]一般是用来标注键值的类型是吗"

**解答**：对。`dict[K, V]` 方括号里第一个是 key 类型，第二个是 value 类型。

但要区分**两种方括号**：

| 写法 | 在哪出现 | 意思 |
|------|----------|------|
| `dict[str, int]` | 类型注解 | 标注"key是str，value是int" |
| `d["age"]` | 实际操作 | 用 key 取值 |

长得一样，意思不同。类型注解的方括号里是**类型**，实际操作的方括号里是**实际的 key 值**。

其他容器类型也用方括号标注：`list[int]`、`tuple[str, int]`、`set[str]`。

**理解状态**：✅ 正确

---

## 二、ToolHandler 与 Callable

### 2.1 ToolHandler 是类型别名是什么意思？

**疑问**："ToolHandler 是类型别名是什么意思"

**解答**：类型别名 = 给一个很长的类型起个短昵称。

```python
# 原始类型，很长
Callable[[dict, dict], tuple[bool, str, dict]]

# 起个别名
ToolHandler = Callable[[dict, dict], tuple[bool, str, dict]]

# 以后用短名
def register(self, tool_def: ToolDef, handler: ToolHandler) -> None:
```

`ToolHandler` 不是新类型，它就是原来那个 `Callable[...]` 的另一个名字。运行时不做任何检查，只是写给人看的。

类比：给"张三"起个外号叫"小张"，后面喊"小张"就是在喊张三。

**理解状态**：✅ 正确（经讲解后理解）

---

### 2.2 Callable 的作用是什么？

**疑问**："callabe作用是什么"

**解答**：`Callable` 是 `typing` 模块里的一个**类型**，表示"可以被调用的东西"（函数、方法、lambda、实现了 `__call__` 的对象）。

它和 `int`、`str`、`dict` 是平级的，都是类型，不是方法。

语法：`Callable[[参数1类型, 参数2类型, ...], 返回值类型]`

```python
Callable[[dict, dict], tuple[bool, str, dict]]
# 两个参数都是 dict，返回三元组
```

**理解状态**：✅ 正确（经讲解后理解）

---

### 2.3 Callable 是一个封装好的方法吗？

**疑问**："所以前面的callable是一个封装好的方法是吗？因为我看后面dict这里没写callabel"

**解答**：❌ 不是。`Callable` 不是方法，是一个**类型**。

它不是"封装好的方法"，不能加括号执行。你不会写 `Callable()` 去调用什么。

dict 那里没写 Callable，是因为用了别名 `ToolHandler`：

```python
ToolHandler = Callable[[dict, dict], tuple[bool, str, dict]]

# 写 ToolHandler 就等于写了完整的 Callable[...]
dict[str, tuple[ToolDef, ToolHandler]]
```

**理解状态**：❌ 理解错误 → 已纠正

---

### 2.4 Callable 是一种特殊的数据类型吗？

**疑问**："就是一种特殊的数据类型是吗"

**解答**：对，是类型的一种，但它描述的不是"存什么数据"，而是"**能不能被调用**"。

| 类型 | 描述什么 | 例子 |
|------|----------|------|
| `int` | 这是个整数（存数据） | `5` |
| `str` | 这是个字符串（存数据） | `"hello"` |
| `dict` | 这是个字典（存数据） | `{"a": 1}` |
| `Callable` | 这是个能加括号执行的东西 | `calculator_handler` |

普通类型描述"存什么"，Callable 描述"能不能跑"。但本质上它和 `int`、`dict` 一样，都是 Python 类型系统里的一个类型。

**理解状态**：✅ 正确

---

### 2.5 ToolHandler 是一个对象吗？

**疑问**："那toolhander是一个对象吗"

**解答**：❌ 不是。`ToolHandler` 是**类型别名**，不是对象。

| 概念 | 是什么 | 例子 |
|------|--------|------|
| `ToolHandler` | 类型（说明书的规格） | "能收两个dict、返回三元组的函数" |
| `calculator_handler` | 对象（具体的函数） | 真正能加括号执行的代码 |

类型不能加括号执行，具体函数才能。就像"整数类型"不能拿来算数，`5` 这个具体整数才能。

**理解状态**：❌ 理解错误 → 已纠正

---

### 2.6 标注了 ToolHandler 就意味着它可以被调用吗？

**疑问**："这样写完就是说这个toolhander可以被调用是吗"

**解答**：❌ 不是。标注不改变对象的行为。

一个东西能不能被调用，取决于它**本身是不是函数**，不取决于你给它标了什么类型。

```python
def calculator_handler(args, context):   # 本身就是函数 → 能调用
    return True, "结果", {}

handler: ToolHandler = calculator_handler  # 标注不影响，还是能调用

x: ToolHandler = "我是字符串"   # 标成 ToolHandler 也不能调用，因为它本身是字符串
x({}, {})                        # ❌ 报错
```

标注就像给人贴名牌写"教师"——名牌不教人教书，只是说明身份。

**理解状态**：❌ 理解错误 → 已纠正

---

### 2.7 ToolHandler 的作用到底是什么？

**疑问**："那这里toolhandler这个对象我还是有些不明白它的作用"

**解答**：`ToolHandler` 是一张**规格说明书**，它不干活。

它规定了："任何被当作工具函数的东西，必须收两个 dict 参数，返回 (bool, str, dict) 三元组。"

真正干活的是 `calculator_handler`、`weather_handler` 这些具体函数。

**为什么需要这张规格说明书？**
1. **IDE 提示**：你传错参数类型时立刻标红
2. **统一接口**：所有工具函数都长一样，Broker 才能用同一行代码调用任何工具：
   ```python
   ok, text, artifacts = handler(args, context)
   ```
   Broker 不需要知道这是计算器还是天气查询，只要符合规格就能统一调用。

类比：ToolHandler 是"厨师上岗证"（规定必须会做菜），calculator_handler 是"一个具体的厨师"（真正做菜）。经理只认上岗证，不认具体是谁。

**理解状态**：✅ 正确（经讲解后理解）

---

### 2.8 函数接收两个参数是因为定义了两个参数吗？

**疑问**："函数是因为它给你定义了两个参数是吗？之所以接收两个参数，是因为定义了两个参数"

**解答**：✅ 对。函数接收几个参数，完全取决于定义时写了几个参数。

```python
def handler(args, context):   # 定义时写了两个 → 调用时必须传两个
    ...
```

`Callable[[dict, dict], ...]` 里的 `[dict, dict]` 就是在说"这个函数应该定义两个参数，都是 dict 类型"。标注和实际定义是一一对应的。

如果实际函数定义的参数个数和标注不一致，IDE 会标红说"类型不匹配"。

**理解状态**：✅ 正确

---

## 三、字典与 self

### 3.1 `self._tools: dict[...] = {}` 是定义了一个字典吗？

**疑问**："self._tools: dict[str, tuple[ToolDef, ToolHandler]] = {}这里是定义了一个字典吗？"

**解答**：✅ 是创建了一个空字典，但要分三层看：

```python
self._tools: dict[str, tuple[ToolDef, ToolHandler]] = {}
#   │              │                                    │
#   │              │                                    └─ 创建空字典（真正运行的部分）
#   │              └─ 类型标注（运行时忽略，给人看）
#   └─ 挂到对象身上
```

`= {}` 才是真正创建字典。冒号后面的类型标注运行时完全忽略。

**理解状态**：✅ 正确

---

### 3.2 内部字典的作用是什么？

**疑问**："内部字典的作用是什么"

**解答**：`__init__` 里有两个字典，是 Broker 对象的"记忆"：

| 字典 | 记住什么 | 没有它会怎样 |
|------|----------|-------------|
| `_tools` | 有哪些工具、每个工具的定义和实现函数 | invoke 时不知道工具存不存在、去哪找函数 |
| `_call_log` | 每个工具最近被调了多少次、什么时间 | 限流功能废掉，LLM 可以无限疯狂调同一工具 |

类比：`_tools` 是员工花名册，`_call_log` 是考勤打卡记录。

**理解状态**：✅ 正确（经讲解后理解）

---

### 3.3 用 self 和不用有什么区别？

**疑问**："用self.tools和不用有什么区别"

**解答**：核心区别是**数据能活多久、归谁管**。

| 方式 | 数据活多久 | 多个对象共享吗 | 适合吗 |
|------|-----------|---------------|--------|
| 局部变量（不用 self） | 函数执行完就销毁 | 不共享（但也用不了） | ❌ 下次调用找不到 |
| 全局变量（不用 self） | 程序结束才销毁 | 所有对象共享 | ❌ 对象之间互相干扰 |
| `self._tools` | 对象活着就一直在 | 每个对象独立一份 | ✅ 既持久又隔离 |

不用 self（局部变量）：register 里存了，register 执行完变量销毁，invoke 里找不到。
不用 self（全局变量）：b1 注册的工具，b2 也能看到，不符合预期。
用 self：对象有记忆，方法之间能共享数据，且每个对象独立。

这就是面向对象的核心：对象 = 状态（属性）+ 行为（方法），方法通过 self 访问自己的状态。

**理解状态**：✅ 正确（经讲解后理解）

---

### 3.4 `self._tools[tool_def.name] = (tool_def, handler)` 是定义字典吗？

**疑问**："self._tools[tool_def.name] = (tool_def, handler)这里也是定义了一个字典是吗，键是名，值是toodef对象和handler函数"

**解答**：⚠️ 用词要纠正。这行**不是定义字典**，是**往已经存在的字典里存一条数据**。

字典是在 `__init__` 里创建的：`self._tools = {}`。
这行是字典赋值操作：

```python
self._tools[tool_def.name] = (tool_def, handler)
#  └──┬───┘  └─────┬─────┘   └──────┬──────┘
#     字典         key（字符串）       value（元组）
```

- key：`tool_def.name`（从 ToolDef 对象取出来的字符串）
- value：`(tool_def, handler)` 元组（第一个是 ToolDef 对象，第二个是函数）

键值的理解✅正确，"定义字典"的用词❌要纠正为"存数据"。

**理解状态**：⚠️ 理解不全（键值正确，用词错误）

---

### 3.5 register 怎么就允许覆盖了？

**疑问**："register这里怎么就允许覆盖了"

**解答**：因为 `self._tools[tool_def.name] = ...` 是 Python dict 的固有行为——key 已存在时，新值直接替换旧值。

```python
d = {}
d["a"] = 1   # 新增
d["a"] = 2   # 覆盖！d["a"] 现在是 2
```

没有写"已存在就报错"的判断，所以就是覆盖。这是**故意的**——支持运行时热更新工具，不用注销再注册。

如果不允许覆盖，就得写成：
```python
if tool_def.name in self._tools:
    raise ValueError("工具已存在")
```

**理解状态**：✅ 正确（经讲解后理解）

---

## 四、ToolDef 与 handler

### 4.1 三个 def 的语法

**疑问**："前面三个def解释一下语法"

**解答**：
```python
def register(self, tool_def: ToolDef, handler: ToolHandler) -> None:
#   │        │     │              │                    │
#   关键字   方法名  self         参数类型标注         返回值标注
```

- `def`：定义函数的关键字
- `self`：对象自己，每个类方法的第一个参数必须是 self
- `tool_def: ToolDef`：参数名 + 类型标注
- `-> None`：返回值类型（None 表示不返回东西）

**理解状态**：✅ 正确（经讲解后理解）

---

### 4.2 为什么有些地方写 name，有些写 .name？

**疑问**："为什么有些地方写name有些.name"

**解答**：`name` 和 `.name` 是两种不同的东西：

| 写法 | 是什么 | 例子 |
|------|--------|------|
| `name` | 一个已经是字符串的变量 | `def unregister(self, name: str)` 里的 name |
| `tool_def.name` | 从对象身上取 name 属性 | `tool_def` 是 ToolDef 对象，`.name` 取它的名字字段 |

- `unregister(name)`、`get(name)`：调用者直接传名字字符串进来，直接用 `name`
- `register(tool_def, handler)`：调用者传的是 ToolDef 对象，要从对象身上取名字，所以写 `tool_def.name`

一句话：**变量直接用，对象后面加点再写属性名。**

**理解状态**：✅ 正确（经讲解后理解）

---

### 4.3 字典的 value 是 `tuple[bool, str, dict]` 吗？

**疑问**："字典的 value 类型里写的是 ToolHandler，不是 Callable，是不是说这个键是str，值可以是tuple[bool, str, dict]"

**解答**：❌ 理解错误。把两层东西搞混了。

字典的 value 是 **`tuple[ToolDef, ToolHandler]`**——一个元组，第一个是 ToolDef 对象，第二个是函数对象。

`tuple[bool, str, dict]` 是**函数执行后的返回值**，是另一层东西：

```python
# 字典里存的是 (工具定义, 函数)
self._tools["calculator"] = (ToolDef(...), calculator_handler)

# 函数执行后才返回 (bool, str, dict)
ok, text, artifacts = calculator_handler(args, context)
# 这才是 tuple[bool, str, dict]
```

| 层级 | 是什么 | 类型 |
|------|--------|------|
| 字典的 value | (工具定义, 函数) 组合 | `tuple[ToolDef, ToolHandler]` |
| 函数执行后的返回值 | (是否成功, 观察文本, 附加信息) | `tuple[bool, str, dict]` |

**理解状态**：❌ 理解错误 → 已纠正

---

### 4.4 字典的值包含 ToolDef 的所有属性吗？

**疑问**："是不是说，这个self属性定义的字典的值应当包含tooldef这个类所定义的所有属性（name, description, parameters, required_role, rate_limit_per_min）"

**解答**：⚠️ 基本正确，但要精确。

不是字典的 value"直接包含"这些属性，而是字典的 value 是一个元组，**元组的第一个元素是 ToolDef 对象，这个对象包含了所有这些属性**。

```python
self._tools = {
    "calculator": (
        ToolDef(name="calculator", description="...", parameters={...}, ...),  # ← 这个对象有所有属性
        calculator_handler
    ),
}
```

访问方式：`self._tools["calculator"][0].name`（先取元组第一个元素，再点属性），不能写成 `self._tools["calculator"]["name"]`（value 是元组不是字典）。

**理解状态**：⚠️ 理解不全（方向正确，访问方式需纠正）

---

### 4.5 ToolDef 是其他模块的类吗？handler 是函数名还是类？

**疑问**："toodef是不是其他模块的类，定义了一些工具，然后handler是具体的函数名还是类"

**解答**：
- **ToolDef**：是 `harness/models.py` 里定义的 Pydantic 类。但它不是"定义了一些工具"，而是"**定义了一个工具的说明书长什么样**"——规定任何工具定义都必须包含 name、description、parameters 等字段。真正的工具定义是 ToolDef 的**实例**（对象）。
- **handler**：是具体的**函数**，不是类。你不会写 `handler()` 创建实例，而是写 `handler(args, context)` 执行它。

一个完整工具 = ToolDef 对象（说明书）+ handler 函数（实现），两者缺一不可。

**理解状态**：✅ 正确（经讲解后理解）

---

## 五、大模型调用工具的完整流程

### 5.1 用户总结的流程（理解完全正确 ✅）

> "大模型看到的是 calculator 工具以及它所需要的一些参数，大模型推理之后告诉我们这些参数具体用值，然后我们把这些值传入函数，就会执行这个工具所代表的函数。"

### 5.2 完整流程（从定义到执行到返回）

1. **定义阶段**：用 Pydantic 定义 `ToolDef` 数据结构，用 `Callable` 起 `ToolHandler` 类型别名，按照约定写出具体工具函数（如 `calculator_handler`）
2. **注册阶段**：把 `ToolDef` 对象和函数一起通过 `register` 注册进 `ToolBroker`，Broker 以工具名为 key、以 `(ToolDef, 函数)` 元组为 value 存进字典
3. **描述生成**：Broker 调用 `list_tool_descriptions` 把所有工具的名字、描述和参数要求翻译成自然语言文本
4. **LLM 推理**：工具描述拼进 Prompt 喂给大模型，大模型看到用户问题和工具清单后推理，输出 `thought`（思考）、`action`（选哪个工具）、`action_input`（参数填什么值）
5. **Broker 执行**：LLM 的决定传给 Broker 的 `invoke` 方法，Broker 用工具名从字典取出实现函数，把参数值作为 `args` 传入，函数执行返回 `(bool, str, dict)`
6. **结果返回**：Broker 把观察文本作为 `observation` 喂回给大模型，大模型基于上一步结果继续推理，决定下一步调用哪个工具还是给出最终答案——如此循环直到任务完成

### 5.3 角色分工

| 角色 | 干什么 | 不干什么 |
|------|--------|----------|
| **LLM** | 看工具描述，决定用哪个工具、填什么参数值 | 不执行工具，只是"出主意" |
| **Broker** | 接收 LLM 的决定，找到对应函数，把参数传进去执行 | 不决定用哪个工具，只是"执行器" |
| **工具函数** | 真正干活（计算、查天气），返回结果 | 不决定什么时候被调用 |

**LLM 是大脑（出主意），Broker 是手（执行），工具函数是具体的工具。**

---

## 六、常见误区纠正清单

| # | 误区 | 纠正 |
|---|------|------|
| 1 | Callable 是封装好的方法 | ❌ Callable 是类型，不是方法，不能加括号执行 |
| 2 | 标注了 ToolHandler 就可以被调用 | ❌ 标注不改变行为，能不能调用取决于本身是不是函数 |
| 3 | ToolHandler 是对象 | ❌ 是类型别名，具体函数才是对象 |
| 4 | 字典 value 是 tuple[bool,str,dict] | ❌ 是 tuple[ToolDef, ToolHandler]；bool/str/dict 是函数返回值 |
| 5 | `self._tools[key] = value` 是定义字典 | ❌ 是往已存在的字典里存数据，字典在 __init__ 里创建 |
| 6 | 类型标注运行时会检查类型 | ❌ 运行时不检查，只是给人看和 IDE 提示 |
| 7 | 不用 self 也能记住注册的工具 | ❌ 局部变量函数执行完就销毁，必须用 self |
| 8 | ToolDef 类定义了一些工具 | ❌ ToolDef 定义了"工具说明书的格式"，具体工具是它的实例 |
| 9 | handler 是类 | ❌ handler 是具体函数，加括号是执行不是创建实例 |

---

## 七、关键结论

1. **类型标注是说明书，不干活**——运行时被忽略，给人看和 IDE 提示用
2. **ToolHandler 是规格，不是对象**——保证所有工具函数签名统一，Broker 才能统一调用
3. **self 让对象有记忆**——方法之间共享数据，且每个对象独立
4. **字典存的是 (说明书, 函数)，函数执行后才返回 (bool, str, dict)**——两层别混
5. **LLM 只负责思考决策，执行全部由确定性代码完成**——这是 ReAct 框架的核心设计思想

---

*文档版本：v1.0*
*整理日期：2026-09-23*
*对应文件：`harness/tool_broker.py`*
