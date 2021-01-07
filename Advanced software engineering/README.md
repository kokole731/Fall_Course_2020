# 说明

## 使用Docker运行项目

  1. 创建Dockerfile文件，并写入相关信息如基本环境、依赖包和执行命令等
  2. 构建Dockerimage，所用命令为`docker build -t asdt .`
  3. 启动Container，所用命令为`docker run asdt`
  4. 不需要手动输入命令执行代码，启动容器后即可输出结果

## 数值微分
### Input
- 初始差 $h_0$  
- 求导点 x  
- 需要求导的函数 $f$  

### Output
- $f$函数在x点的导数值

### Two cases:
 - Case 1: Init h0: 0.01, function: add1, point: 1   ---> 1
 - Case 2: Init h0: 0.01, function: square, point: 1    ---> 2.015625
   
***
Ref: [Why functional programming matters](https://www.researchgate.net/publication/2452204_Why_Functional_Programming_Matters)