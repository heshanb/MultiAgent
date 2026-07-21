import ezdxf

# 创建DXF文档
doc = ezdxf.new(dxfversion="R2018", units=4)  # 4 = MM (INSUNITS)
msp = doc.modelspace()

# 画一个阶梯轴轮廓
# 外轮廓矩形
msp.add_lwpolyline([(0, 0), (100, 0), (100, 40), (0, 40), (0, 0)])
# 中间台阶
msp.add_lwpolyline([(30, 0), (30, 25), (70, 25), (70, 0)])

# 添加尺寸标注（关键，用来测试尺寸提取）
# 总长100mm
dim1 = msp.add_linear_dim(base=(0, -10), p1=(0, 0), p2=(100, 0))
dim1.render()
# 台阶长度40mm
dim2 = msp.add_linear_dim(base=(50, -20), p1=(30, 0), p2=(70, 0))
dim2.render()
# 总高40
dim3 = msp.add_linear_dim(base=(-10, 20), p1=(0, 0), p2=(0, 40), angle=90)
dim3.render()

# 添加图层
doc.layers.new("尺寸层", dxfattribs={"color": 1})
doc.layers.new("轮廓层", dxfattribs={"color": 7})

# 保存文件
doc.saveas("part.dxf")
print("测试图纸 part.dxf 生成完成！")