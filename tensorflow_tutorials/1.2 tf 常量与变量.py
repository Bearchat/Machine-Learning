import tensorflow as tf

## 定义变量  常量

# 定义一个变量,用 tf.Variable 定义变量，与python不同的是，必须先定义它是一个变量，它才是一个变量，初始值为0，还可以给它一个名字
# 在初始化 init = tf.global_variables_initializer()  之前 所有Variable变量的值都是默认值  0
var = tf.Variable(0, name="myvar")

# 将 new_value 加载到 var 上，myvar就被更新了
update_var = tf.assign(var, 10)

# 定义一个常量
con_var = tf.constant(1)

# 定义一个加法
new_var = tf.add(update_var, con_var)

## 开始计算

print('before :', var)
# 初始化，在初始化之前是变量是没有值的
init = tf.global_variables_initializer()
print('after :', var)

# 这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步.
sess = tf.Session()

# 计算
sess.run(init)

# 输出
print('var : ', sess.run(var))
print('con_var : ', sess.run(con_var))
print('new_var : ', sess.run(new_var))

# 关闭会话
sess.close()

"""
# 另一种写法
with tf.Session() as sess:
    sess.run(init)
    print ('var : ',sess.run(var))
    print ('con_var : ',sess.run(con_var))
    print ('new_var : ',sess.run(new_var))
"""
