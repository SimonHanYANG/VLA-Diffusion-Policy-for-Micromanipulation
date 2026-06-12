# Simulator 重新设计总结

## 完成的修改

### 1. 背景生成器 (`src/simulator/background.py`)

**新增 `SingleImageBackground` 类：**
- 使用单张真实显微镜背景图片（默认 `data/backgrounds_test/bg_00000.png`）
- 支持随机裁剪（如果图片大于目标尺寸）
- 支持随机翻转增强
- 支持亮度调整（`brightness_range` 参数，默认 0.8-1.2）
- 图片只加载一次并缓存，提高性能

**使用方法：**
```python
from src.simulator.background import SingleImageBackground

bg = SingleImageBackground(
    image_path=Path("data/backgrounds_test/bg_00000.png"),
    brightness_range=(0.8, 1.2),
)
```

### 2. 精子生成器 (`src/simulator/targets.py`)

**新增 `SpermGenerator` 类：**
- 生成完整精子形状：椭圆形头部 + Bezier 曲线尾部
- 头部位于尾部起点（p0），形成完整的精子形态
- 头部是椭圆形，带有尖锐的前端（指向尾部方向）
- 尾部是 Bezier 曲线，从头部延伸出去
- 参考点是头部中心（p0位置）

**特性：**
- 头部纹理：较暗（55-80），带有 Acrosome 亮点
- 尾部纹理：较亮（120-140），几乎透明
- 支持随机化：头部大小、尾部长度、曲线形状等

**使用方法：**
```python
from src.simulator.targets import SpermGenerator

gen = SpermGenerator()
target = gen.generate(image_size=(224, 224), rng=rng)
# target.label = "sperm"
# target.mask: 完整精子的 mask
# target.texture: 完整精子的纹理
# target.reference_point: 头部中心位置
```

### 3. 渲染器修改 (`src/simulator/renderer.py`)

**修改 `MicroscopeRenderer.render()` 方法：**
- 所有图像处理在灰度图上进行
- 最后转换为 3 通道（灰度 × 3）
- 保持灰度一致性，避免颜色伪影

**输出格式：**
- 形状：(H, W, 3)
- 数据类型：uint8
- 通道：灰度 × 3（所有通道相同）

### 4. 配置更新 (`src/utils/config.py`)

**在 `SimulatorConfig` 中添加：**
```python
background_type: str = "single_image"  # "perlin", "image", "single_image"
background_image: str = "data/backgrounds_test/bg_00000.png"
background_brightness_range: Tuple[float, float] = (0.8, 1.2)
```

### 5. 环境更新 (`src/simulator/environment.py`)

**修改 `MicroscopeEnvironment.__init__()`：**
- 根据 `config.background_type` 自动选择背景生成器
- 支持三种背景类型：`"perlin"`, `"image"`, `"single_image"`

## 测试结果

### 单元测试

所有 41 个测试通过：
```
tests/test_simulator.py::TestStageSimulator ✓ (6 tests)
tests/test_simulator.py::TestTargetGenerators ✓ (8 tests)
tests/test_simulator.py::TestNoiseApplicator ✓ (4 tests)
tests/test_simulator.py::TestPerlinNoiseBackground ✓ (4 tests)
tests/test_simulator.py::TestMicroscopeEnvironment ✓ (7 tests)
tests/test_simulator.py::TestAllTaskEnvironments ✓ (4 tests)
tests/test_simulator.py::TestSpermGenerator ✓ (4 tests) [NEW]
tests/test_simulator.py::TestSingleImageBackground ✓ (4 tests) [NEW]
```

### 功能验证

运行 `scripts/test_new_simulator.py` 生成测试样本：
- ✅ 小球（microsphere）正常生成
- ✅ 酵母菌（yeast）正常生成
- ✅ 完整精子（sperm）正常生成（头部 + 尾部）
- ✅ 真实背景图片正常加载
- ✅ 亮度调整正常工作
- ✅ 环境渲染正常（输出 3 通道灰度图）

## 使用示例

### 1. 使用真实背景生成轨迹

```python
from pathlib import Path
from src.simulator.targets import SpermGenerator
from src.simulator.background import SingleImageBackground
from src.simulator.environment import MicroscopeEnvironment
from src.utils.config import SimulatorConfig

# 配置
config = SimulatorConfig(
    image_size=(224, 224),
    background_type="single_image",
    background_image="data/backgrounds_test/bg_00000.png",
    background_brightness_range=(0.8, 1.2),
)

# 创建目标生成器
gen = SpermGenerator()

# 创建环境
env = MicroscopeEnvironment(config=config, target_generator=gen)

# 生成轨迹
obs, info = env.reset(seed=42)
for step in range(10):
    action = np.array([5.0, -3.0])
    obs, reward, done, info = env.step(action)
    if done:
        break
```

### 2. 使用工厂函数

```python
from src.simulator.targets import get_target_generator_for_task

# 获取各种目标生成器
microsphere_gen = get_target_generator_for_task("microsphere")
yeast_gen = get_target_generator_for_task("yeast")
sperm_gen = get_target_generator_for_task("sperm")  # 新增
```

### 3. 通过配置文件

```yaml
# config.yaml
image_size: [224, 224]
background_type: "single_image"
background_image: "data/backgrounds_test/bg_00000.png"
background_brightness_range: [0.8, 1.2]
```

## 文件清单

### 修改的文件
1. `src/simulator/background.py` - 新增 SingleImageBackground
2. `src/simulator/targets.py` - 新增 SpermGenerator
3. `src/simulator/renderer.py` - 修改为灰度 ×3 输出
4. `src/simulator/environment.py` - 支持新配置
5. `src/utils/config.py` - 添加背景配置
6. `tests/test_simulator.py` - 添加新测试

### 新增的文件
1. `scripts/test_new_simulator.py` - 测试脚本
2. `docs/simulator_redesign_summary.md` - 本文档

### 生成的测试文件
1. `data/test_new_simulator/microsphere_mask.png`
2. `data/test_new_simulator/microsphere_texture.png`
3. `data/test_new_simulator/yeast_mask.png`
4. `data/test_new_simulator/yeast_texture.png`
5. `data/test_new_simulator/sperm_mask.png`
6. `data/test_new_simulator/sperm_texture.png`
7. `data/test_new_simulator/env_microsphere_frame0.png`
8. `data/test_new_simulator/env_yeast_frame0.png`
9. `data/test_new_simulator/env_sperm_frame0.png`
10. ... (更多帧)

## 向后兼容性

- ✅ 保留了原有的 `SpermHeadGenerator` 和 `SpermTailGenerator`
- ✅ 保留了原有的 `PerlinNoiseBackground` 和 `ImageBackground`
- ✅ 所有现有测试继续通过
- ✅ 配置文件向后兼容（新字段有默认值）

## 下一步建议

1. **生成训练数据**：使用新的 simulator 生成大量训练轨迹
2. **可视化验证**：检查生成的图像是否符合预期
3. **调整参数**：根据需要调整亮度范围、头部大小等参数
4. **批量生成**：使用 `scripts/generate_expert_data.py` 生成训练数据集

## 性能说明

- 背景图片只加载一次并缓存，后续调用直接使用缓存
- 精子生成器使用向量化操作，性能良好
- 灰度 ×3 转换使用 numpy stack，效率高
