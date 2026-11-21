
编号	实验名称	消融内容
01	Full	完整模型，作为对照组
02	No GMLS-v2 Correction	关闭 GMLS-v2 修正分支
03	No Final Center	关闭 GeoCenter V2 最终中心损失
04	Top-1 ROI	Top-2 改为 Top-1
05	One Head	双头改为单共享检测头，保留全局辅助监督
06	No Global Aux	保留双头，关闭全局辅助损失
07	Local Only	单头，仅进行局部检测监督
08	No Top-2 Diverse	关闭推理阶段 Top-2 多样化选择
