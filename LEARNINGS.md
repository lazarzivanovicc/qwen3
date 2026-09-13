# RoPE as always :D

I still have some confusion - I get it that based on the index position of a component of a vector and based on the position of a vector of a token I will create an angle. So creation of those angles is not really a mistery to me, but the way that application algorithm works is. Check it out and decompose it to the fullest extent!

With column vectors -> A_rot * x = x_rot but with row vectors x_t * A_rot_t = x_rot_t

x_t = [x1 x2], A_rot_t = [[cos_theta, sin_theta], [-sin_theta, cos_theta]] => x_rot_t = [x1 * cos_theta - x2 * sin_theta, x1 * sin_theta + x2 * cos_theta] ... (1)

==========

In RoPE
x = [x1, x2, x3, x4]
rotated = [-x3, -x4, x1, x2]
cos = [cos(theta_1), cos(theta_2), cos(theta_1), cos(theta_2)]
sin = [sin(theta_1), sin(theta_2), sin(theta_1), sin(theta_2)]
x_rotated = (x * cos) + (rotated * sin)
x_rotated = (x1 * cos(theta_1), x2 * cos(theta_2), x3 * cos(theta_1), x4 * cos(theta_2)) + (-x3 * sin(theta_1), -x4 * sin(theta_2), x1 * sin(theta_1), x2 * sin(theta_2))
x_rotated = (x1 * cos(theta_1) - x3 * sin(theta_1), x2 * cos(theta_2) - x4 * sin(theta_2), x3 * cos(theta_1) + x1 * sin(theta_1), x4 * cos(theta_2) + x2 * sin(theta_2))

Za isti ugao theta rotiramo i i i + d // 2 koordinatu x vektora i ovi parovi odgovaraju 2D rotaciji koju smo prikazali u ...(1)
Prvih d//2 koordinata x_rotated vektora dobijamo po x1 * cos(theta_1) - x3 * sin(theta_1) receptu, dok drugih d // 2 dobijamo kao x3 * cos(theta_1) + x1 * sin(theta_1)

# KV cache - let's wrestle with it

One KV cache - accessed by all the attention layers.

Simple dict with K and V keys and each key has 28 empty tensors as values

During the prefil we calculate K and V values (apply rotation) and save them

In every subsequent pass we must change apply_rope fn to access the position of the cos and sin tables as I was only accessing the absolute position of the last token

After that I take the values stored in KV cache based on the layer and append new k and v values in dim 2 (sequence dimension), after which I set k and q values to be those new values from cache

Mask has to be adjusted we will see how exectly! When calculating prefix mask is normal causal mask - in every subsequent pass - new token generation mask is only one row with total_sequence (prefil + current_pos) long and is all False after which it does not actually mask anything when we use masked_fill

# Creating one mask and passing it to other layers

Qwen3 uses Causal Self Attention ofc, but I tried to create a buffer like I did with GPT2 and I failed because it required me to create biggere matrices (4096, 4096) - max context length - 28 times and I think I got OOM

# Streaming 

I want to print out text immeditely.

Generators and Yield are the answer - check scratchpad to better understand this.

# Chat template
<|im_start|>user\nGive me a short introduction to large language models.<|im_end|>\n<|im_start|>assistant\n>

# Top P, Top K

# Model weights are in BF-16

1 sign bit, 8 bits for exponent (E) and 7 bits for mantissa

How will i convert this to FP32?

(-1)^sign bit * 2^(E - 2^7 - 1) * 1.mantissa

ako su 1110010 bitovi mantise u bf16

Onda je u float32 ova mantisa 111001000000000000000 (nisam siguran da sam ubo broj nula - treba da ih ima 23-7=16 novih nula)

U Javi da bih od shorta napravio 32 bitni podatak short & 0xFFFF - sacuvam orignalnih 16 podataka pa dodam 16 nula na pocetak - 0000 0000 0000 0000 originalni broj - i sada ga left siftujem << 16 i dobijem
orignalni broj 0000 0000 0000 0000

