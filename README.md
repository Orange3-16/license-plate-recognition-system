# 面向雨雾天气的车牌识别系统

##项目结构

### 一、核心代码模块
```
├── 📂 data/                    # 数据处理模块
├── 📂 detection/               # 车牌检测模块
├── 📂 dehaze/                  # 图像去雾模块  
├── 📂 recognition/             # 字符识别模块
└──  📂 runs/                    # 训练结果和模型文件
```
### 二、主要脚本文件
```
├── 📄 demo.py                  # 系统演示脚本
└── 📄 requirements.txt         # 项目依赖
```
## 数据来源

本项目使用[CCPD数据集](https://github.com/detectRecog/CCPD)进行训练和验证。该数据集包含约30万张中国车牌图像，覆盖多种天气和光照条件。

### 数据使用声明
- 数据集遵循CC BY-NC-SA 4.0许可证
- 仅用于学术研究目的
- 详细的数据处理流程请参考[DATA_SOURCES.md](DATA_SOURCES.md)

### 引用要求
如果您在研究中使用了本项目，请引用原始CCPD论文：

```bibtex
@inproceedings{xu2018ccpd,
  title={CCPD: A diverse and large-scale dataset for license plate detection and recognition},
  author={Xu, Zhenbo and Yang, Wei and Meng, Ajing and Lu, Nanxue and Huang, Huan and Ying, Chenping and Huang, Liusheng},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={--},
  year={2018}
}
```
```

### 3. 创建许可证文件 `LICENSE_DATA.md`

```markdown
# 数据使用许可证声明

## 项目代码许可证
本项目代码采用 **MIT 许可证**，详见 [LICENSE](LICENSE) 文件。

## 数据集许可证
本项目使用的CCPD数据集遵循 **CC BY-NC-SA 4.0** 许可证。

### CC BY-NC-SA 4.0 主要条款
- **署名 (BY)**: 必须给出适当的署名
- **非商业性使用 (NC)**: 不得将材料用于商业目的
- **相同方式共享 (SA)**: 如果改编、转换或以本材料为基础创作，必须按相同许可证分发作品

### 使用限制
1. **禁止商业使用**: 不得将数据集或基于数据集的结果用于商业目的
2. **署名要求**: 使用数据集时必须引用原始论文
3. **共享要求**: 基于数据集的新作品必须采用相同许可证

## 合规使用建议

### 学术研究
- 可以自由使用本项目进行学术研究
- 必须在论文中引用CCPD原始论文
- 研究成果可以公开发表

### 商业应用
- **禁止**直接使用本项目进行商业应用
- 如需商业使用，请获取原始数据集的商业许可证
- 或使用其他允许商业使用的数据集

## 法律声明

本项目开发者不对数据集的合规使用承担法律责任。使用者应自行确保遵守相关许可证条款。

## 联系方式

如有关于数据使用的疑问，请联系项目维护者或参考原始数据集官方页面。
```
