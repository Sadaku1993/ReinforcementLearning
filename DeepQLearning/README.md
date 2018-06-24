# Deep Q Learning

<dl>
    <dt>tf.assign</dt>
    <dd>Variableに特定の値を代入する際に使用する</dd>
</dl>

ターゲットQネットワークの重み更新の関数サンプル
```
def update_target_q_weights(self):
    target_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network"
    )
    predict_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="predict_network"
    )
    for target_var, predict_var in zip(target_vars, predict_vars):
        weight_init = tf.placeholder("float32", name="weight")
        targt_var.assign(weight_input).eval(
            {weight_input: predict_var.eval()}
        )
```
