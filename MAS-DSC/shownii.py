import torchio as tio
t1_path = './Data/BraTS19_2013_2_1_t1.nii.gz'#读取T1的nii文件
t1_img = tio.ScalarImage(t1_path)#将nii文件转换为tio.ScalarImage格式
t1_img.plot()
print(t1_img.shape)#(240, 240, 155)
