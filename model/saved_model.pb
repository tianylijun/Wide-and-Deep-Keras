ЩД
ф§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8▓н
ё
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

:	*
dtype0
ѕ
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_1/embeddings
Ђ
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:*
dtype0
ѕ
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_2/embeddings
Ђ
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes

:*
dtype0
ѕ
embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_3/embeddings
Ђ
*embedding_3/embeddings/Read/ReadVariableOpReadVariableOpembedding_3/embeddings*
_output_shapes

:*
dtype0
ѕ
embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_4/embeddings
Ђ
*embedding_4/embeddings/Read/ReadVariableOpReadVariableOpembedding_4/embeddings*
_output_shapes

:*
dtype0
ѕ
embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_5/embeddings
Ђ
*embedding_5/embeddings/Read/ReadVariableOpReadVariableOpembedding_5/embeddings*
_output_shapes

:*
dtype0
ѕ
embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_6/embeddings
Ђ
*embedding_6/embeddings/Read/ReadVariableOpReadVariableOpembedding_6/embeddings*
_output_shapes

:*
dtype0
ѕ
embedding_7/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**'
shared_nameembedding_7/embeddings
Ђ
*embedding_7/embeddings/Read/ReadVariableOpReadVariableOpembedding_7/embeddings*
_output_shapes

:**
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D2*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:D2*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:2*
dtype0
r
deep/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedeep/kernel
k
deep/kernel/Read/ReadVariableOpReadVariableOpdeep/kernel*
_output_shapes

:2*
dtype0
j
	deep/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	deep/bias
c
deep/bias/Read/ReadVariableOpReadVariableOp	deep/bias*
_output_shapes
:*
dtype0
}
wide_deep/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј*!
shared_namewide_deep/kernel
v
$wide_deep/kernel/Read/ReadVariableOpReadVariableOpwide_deep/kernel*
_output_shapes
:	ј*
dtype0
t
wide_deep/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namewide_deep/bias
m
"wide_deep/bias/Read/ReadVariableOpReadVariableOpwide_deep/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
њ
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*,
shared_nameAdam/embedding/embeddings/m
І
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes

:	*
dtype0
ќ
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_1/embeddings/m
Ј
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes

:*
dtype0
ќ
Adam/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_2/embeddings/m
Ј
1Adam/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/m*
_output_shapes

:*
dtype0
ќ
Adam/embedding_3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_3/embeddings/m
Ј
1Adam/embedding_3/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_3/embeddings/m*
_output_shapes

:*
dtype0
ќ
Adam/embedding_4/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_4/embeddings/m
Ј
1Adam/embedding_4/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_4/embeddings/m*
_output_shapes

:*
dtype0
ќ
Adam/embedding_5/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_5/embeddings/m
Ј
1Adam/embedding_5/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_5/embeddings/m*
_output_shapes

:*
dtype0
ќ
Adam/embedding_6/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_6/embeddings/m
Ј
1Adam/embedding_6/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_6/embeddings/m*
_output_shapes

:*
dtype0
ќ
Adam/embedding_7/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**.
shared_nameAdam/embedding_7/embeddings/m
Ј
1Adam/embedding_7/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_7/embeddings/m*
_output_shapes

:**
dtype0
ѓ
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D2*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:D2*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:2*
dtype0
ђ
Adam/deep/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*#
shared_nameAdam/deep/kernel/m
y
&Adam/deep/kernel/m/Read/ReadVariableOpReadVariableOpAdam/deep/kernel/m*
_output_shapes

:2*
dtype0
x
Adam/deep/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/deep/bias/m
q
$Adam/deep/bias/m/Read/ReadVariableOpReadVariableOpAdam/deep/bias/m*
_output_shapes
:*
dtype0
І
Adam/wide_deep/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј*(
shared_nameAdam/wide_deep/kernel/m
ё
+Adam/wide_deep/kernel/m/Read/ReadVariableOpReadVariableOpAdam/wide_deep/kernel/m*
_output_shapes
:	ј*
dtype0
ѓ
Adam/wide_deep/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/wide_deep/bias/m
{
)Adam/wide_deep/bias/m/Read/ReadVariableOpReadVariableOpAdam/wide_deep/bias/m*
_output_shapes
:*
dtype0
њ
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*,
shared_nameAdam/embedding/embeddings/v
І
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes

:	*
dtype0
ќ
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_1/embeddings/v
Ј
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes

:*
dtype0
ќ
Adam/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_2/embeddings/v
Ј
1Adam/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/v*
_output_shapes

:*
dtype0
ќ
Adam/embedding_3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_3/embeddings/v
Ј
1Adam/embedding_3/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_3/embeddings/v*
_output_shapes

:*
dtype0
ќ
Adam/embedding_4/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_4/embeddings/v
Ј
1Adam/embedding_4/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_4/embeddings/v*
_output_shapes

:*
dtype0
ќ
Adam/embedding_5/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_5/embeddings/v
Ј
1Adam/embedding_5/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_5/embeddings/v*
_output_shapes

:*
dtype0
ќ
Adam/embedding_6/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/embedding_6/embeddings/v
Ј
1Adam/embedding_6/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_6/embeddings/v*
_output_shapes

:*
dtype0
ќ
Adam/embedding_7/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**.
shared_nameAdam/embedding_7/embeddings/v
Ј
1Adam/embedding_7/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_7/embeddings/v*
_output_shapes

:**
dtype0
ѓ
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:D2*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:D2*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:2*
dtype0
ђ
Adam/deep/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*#
shared_nameAdam/deep/kernel/v
y
&Adam/deep/kernel/v/Read/ReadVariableOpReadVariableOpAdam/deep/kernel/v*
_output_shapes

:2*
dtype0
x
Adam/deep/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/deep/bias/v
q
$Adam/deep/bias/v/Read/ReadVariableOpReadVariableOpAdam/deep/bias/v*
_output_shapes
:*
dtype0
І
Adam/wide_deep/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј*(
shared_nameAdam/wide_deep/kernel/v
ё
+Adam/wide_deep/kernel/v/Read/ReadVariableOpReadVariableOpAdam/wide_deep/kernel/v*
_output_shapes
:	ј*
dtype0
ѓ
Adam/wide_deep/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/wide_deep/bias/v
{
)Adam/wide_deep/bias/v/Read/ReadVariableOpReadVariableOpAdam/wide_deep/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Эm
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*│m
valueЕmBдm BЪm
н
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer_with_weights-4
layer-16
layer_with_weights-5
layer-17
layer_with_weights-6
layer-18
layer_with_weights-7
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer_with_weights-8
layer-26
layer-27
layer_with_weights-9
layer-28
layer-29
layer-30
 layer-31
!layer_with_weights-10
!layer-32
"	optimizer
#	variables
$regularization_losses
%trainable_variables
&	keras_api
'
signatures
 
 
 
 
 
 
 
 
 
 
 
 
b
(
embeddings
)	variables
*regularization_losses
+trainable_variables
,	keras_api
b
-
embeddings
.	variables
/regularization_losses
0trainable_variables
1	keras_api
b
2
embeddings
3	variables
4regularization_losses
5trainable_variables
6	keras_api
b
7
embeddings
8	variables
9regularization_losses
:trainable_variables
;	keras_api
b
<
embeddings
=	variables
>regularization_losses
?trainable_variables
@	keras_api
b
A
embeddings
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
b
F
embeddings
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
b
K
embeddings
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
R
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
R
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
R
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
R
\	variables
]regularization_losses
^trainable_variables
_	keras_api
R
`	variables
aregularization_losses
btrainable_variables
c	keras_api
R
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
h

hkernel
ibias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
R
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
h

rkernel
sbias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
 
R
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
R
|	variables
}regularization_losses
~trainable_variables
	keras_api
n
ђkernel
	Ђbias
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
Ё	keras_api
р
	єiter
Єbeta_1
ѕbeta_2

Ѕdecay
іlearning_rate(m -mђ2mЂ7mѓ<mЃAmёFmЁKmєhmЄimѕrmЅsmі	ђmІ	Ђmї(vЇ-vј2vЈ7vљ<vЉAvњFvЊKvћhvЋivќrvЌsvў	ђvЎ	Ђvџ
h
(0
-1
22
73
<4
A5
F6
K7
h8
i9
r10
s11
ђ12
Ђ13
 
h
(0
-1
22
73
<4
A5
F6
K7
h8
i9
r10
s11
ђ12
Ђ13
▓
Іmetrics
 їlayer_regularization_losses
#	variables
Їlayers
јnon_trainable_variables
Јlayer_metrics
$regularization_losses
%trainable_variables
 
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

(0
 

(0
▓
љmetrics
 Љlayer_regularization_losses
)	variables
њlayers
Њnon_trainable_variables
ћlayer_metrics
*regularization_losses
+trainable_variables
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

-0
 

-0
▓
Ћmetrics
 ќlayer_regularization_losses
.	variables
Ќlayers
ўnon_trainable_variables
Ўlayer_metrics
/regularization_losses
0trainable_variables
fd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE

20
 

20
▓
џmetrics
 Џlayer_regularization_losses
3	variables
юlayers
Юnon_trainable_variables
ъlayer_metrics
4regularization_losses
5trainable_variables
fd
VARIABLE_VALUEembedding_3/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE

70
 

70
▓
Ъmetrics
 аlayer_regularization_losses
8	variables
Аlayers
бnon_trainable_variables
Бlayer_metrics
9regularization_losses
:trainable_variables
fd
VARIABLE_VALUEembedding_4/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE

<0
 

<0
▓
цmetrics
 Цlayer_regularization_losses
=	variables
дlayers
Дnon_trainable_variables
еlayer_metrics
>regularization_losses
?trainable_variables
fd
VARIABLE_VALUEembedding_5/embeddings:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUE

A0
 

A0
▓
Еmetrics
 фlayer_regularization_losses
B	variables
Фlayers
гnon_trainable_variables
Гlayer_metrics
Cregularization_losses
Dtrainable_variables
fd
VARIABLE_VALUEembedding_6/embeddings:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUE

F0
 

F0
▓
«metrics
 »layer_regularization_losses
G	variables
░layers
▒non_trainable_variables
▓layer_metrics
Hregularization_losses
Itrainable_variables
fd
VARIABLE_VALUEembedding_7/embeddings:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUE

K0
 

K0
▓
│metrics
 ┤layer_regularization_losses
L	variables
хlayers
Хnon_trainable_variables
иlayer_metrics
Mregularization_losses
Ntrainable_variables
 
 
 
▓
Иmetrics
 ╣layer_regularization_losses
P	variables
║layers
╗non_trainable_variables
╝layer_metrics
Qregularization_losses
Rtrainable_variables
 
 
 
▓
йmetrics
 Йlayer_regularization_losses
T	variables
┐layers
└non_trainable_variables
┴layer_metrics
Uregularization_losses
Vtrainable_variables
 
 
 
▓
┬metrics
 ├layer_regularization_losses
X	variables
─layers
┼non_trainable_variables
кlayer_metrics
Yregularization_losses
Ztrainable_variables
 
 
 
▓
Кmetrics
 ╚layer_regularization_losses
\	variables
╔layers
╩non_trainable_variables
╦layer_metrics
]regularization_losses
^trainable_variables
 
 
 
▓
╠metrics
 ═layer_regularization_losses
`	variables
╬layers
¤non_trainable_variables
лlayer_metrics
aregularization_losses
btrainable_variables
 
 
 
▓
Лmetrics
 мlayer_regularization_losses
d	variables
Мlayers
нnon_trainable_variables
Нlayer_metrics
eregularization_losses
ftrainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1
 

h0
i1
▓
оmetrics
 Оlayer_regularization_losses
j	variables
пlayers
┘non_trainable_variables
┌layer_metrics
kregularization_losses
ltrainable_variables
 
 
 
▓
█metrics
 ▄layer_regularization_losses
n	variables
Пlayers
яnon_trainable_variables
▀layer_metrics
oregularization_losses
ptrainable_variables
WU
VARIABLE_VALUEdeep/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	deep/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

r0
s1
 

r0
s1
▓
Яmetrics
 рlayer_regularization_losses
t	variables
Рlayers
сnon_trainable_variables
Сlayer_metrics
uregularization_losses
vtrainable_variables
 
 
 
▓
тmetrics
 Тlayer_regularization_losses
x	variables
уlayers
Уnon_trainable_variables
жlayer_metrics
yregularization_losses
ztrainable_variables
 
 
 
▓
Жmetrics
 вlayer_regularization_losses
|	variables
Вlayers
ьnon_trainable_variables
Ьlayer_metrics
}regularization_losses
~trainable_variables
][
VARIABLE_VALUEwide_deep/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEwide_deep/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

ђ0
Ђ1
 

ђ0
Ђ1
х
№metrics
 ­layer_regularization_losses
ѓ	variables
ыlayers
Ыnon_trainable_variables
зlayer_metrics
Ѓregularization_losses
ёtrainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

З0
ш1
 
■
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Шtotal

эcount
Э	variables
щ	keras_api
I

Щtotal

чcount
Ч
_fn_kwargs
§	variables
■	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ш0
э1

Э	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Щ0
ч1

§	variables
ѕЁ
VARIABLE_VALUEAdam/embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_1/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_2/embeddings/mVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_3/embeddings/mVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_4/embeddings/mVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_5/embeddings/mVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_6/embeddings/mVlayer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_7/embeddings/mVlayer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/deep/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/deep/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/wide_deep/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/wide_deep/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUEAdam/embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_1/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_2/embeddings/vVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_3/embeddings/vVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_4/embeddings/vVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_5/embeddings/vVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_6/embeddings/vVlayer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUEAdam/embedding_7/embeddings/vVlayer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/deep/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/deep/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/wide_deep/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/wide_deep/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
y
serving_default_age_inPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
ѓ
serving_default_capital_gain_inPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
ѓ
serving_default_capital_loss_inPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
ђ
serving_default_education_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
}
serving_default_gender_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
ё
!serving_default_hours_per_week_inPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Ё
"serving_default_marital_status_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Ё
"serving_default_native_country_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Ђ
serving_default_occupation_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_race_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Ѓ
 serving_default_relationship_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
y
serving_default_widePlaceholder*(
_output_shapes
:         Щ*
dtype0*
shape:         Щ
ђ
serving_default_workclass_inpPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
┴
StatefulPartitionedCallStatefulPartitionedCallserving_default_age_inserving_default_capital_gain_inserving_default_capital_loss_inserving_default_education_inpserving_default_gender_inp!serving_default_hours_per_week_in"serving_default_marital_status_inp"serving_default_native_country_inpserving_default_occupation_inpserving_default_race_inp serving_default_relationship_inpserving_default_wideserving_default_workclass_inpembedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsembedding_3/embeddingsembedding_4/embeddingsembedding_5/embeddingsembedding_6/embeddingsembedding_7/embeddingsdense/kernel
dense/biasdeep/kernel	deep/biaswide_deep/kernelwide_deep/bias*&
Tin
2*
Tout
2*'
_output_shapes
:         *0
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference_signature_wrapper_21456
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
є
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp*embedding_2/embeddings/Read/ReadVariableOp*embedding_3/embeddings/Read/ReadVariableOp*embedding_4/embeddings/Read/ReadVariableOp*embedding_5/embeddings/Read/ReadVariableOp*embedding_6/embeddings/Read/ReadVariableOp*embedding_7/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpdeep/kernel/Read/ReadVariableOpdeep/bias/Read/ReadVariableOp$wide_deep/kernel/Read/ReadVariableOp"wide_deep/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp1Adam/embedding_2/embeddings/m/Read/ReadVariableOp1Adam/embedding_3/embeddings/m/Read/ReadVariableOp1Adam/embedding_4/embeddings/m/Read/ReadVariableOp1Adam/embedding_5/embeddings/m/Read/ReadVariableOp1Adam/embedding_6/embeddings/m/Read/ReadVariableOp1Adam/embedding_7/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp&Adam/deep/kernel/m/Read/ReadVariableOp$Adam/deep/bias/m/Read/ReadVariableOp+Adam/wide_deep/kernel/m/Read/ReadVariableOp)Adam/wide_deep/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp1Adam/embedding_2/embeddings/v/Read/ReadVariableOp1Adam/embedding_3/embeddings/v/Read/ReadVariableOp1Adam/embedding_4/embeddings/v/Read/ReadVariableOp1Adam/embedding_5/embeddings/v/Read/ReadVariableOp1Adam/embedding_6/embeddings/v/Read/ReadVariableOp1Adam/embedding_7/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp&Adam/deep/kernel/v/Read/ReadVariableOp$Adam/deep/bias/v/Read/ReadVariableOp+Adam/wide_deep/kernel/v/Read/ReadVariableOp)Adam/wide_deep/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_22805
Ё
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsembedding_3/embeddingsembedding_4/embeddingsembedding_5/embeddingsembedding_6/embeddingsembedding_7/embeddingsdense/kernel
dense/biasdeep/kernel	deep/biaswide_deep/kernelwide_deep/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/embedding/embeddings/mAdam/embedding_1/embeddings/mAdam/embedding_2/embeddings/mAdam/embedding_3/embeddings/mAdam/embedding_4/embeddings/mAdam/embedding_5/embeddings/mAdam/embedding_6/embeddings/mAdam/embedding_7/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/deep/kernel/mAdam/deep/bias/mAdam/wide_deep/kernel/mAdam/wide_deep/bias/mAdam/embedding/embeddings/vAdam/embedding_1/embeddings/vAdam/embedding_2/embeddings/vAdam/embedding_3/embeddings/vAdam/embedding_4/embeddings/vAdam/embedding_5/embeddings/vAdam/embedding_6/embeddings/vAdam/embedding_7/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/deep/kernel/vAdam/deep/bias/vAdam/wide_deep/kernel/vAdam/wide_deep/bias/v*?
Tin8
624*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_restore_22970ч╦
З
C
'__inference_reshape_layer_call_fn_22234

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204292
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Яџ
┐
 __inference__wrapped_model_20168
wide
workclass_inp
education_inp
marital_status_inp
occupation_inp
relationship_inp
race_inp

gender_inp
native_country_inp

age_in
capital_gain_in
capital_loss_in
hours_per_week_in*
&model_embedding_embedding_lookup_20062,
(model_embedding_1_embedding_lookup_20067,
(model_embedding_2_embedding_lookup_20072,
(model_embedding_3_embedding_lookup_20077,
(model_embedding_4_embedding_lookup_20082,
(model_embedding_5_embedding_lookup_20087,
(model_embedding_6_embedding_lookup_20092,
(model_embedding_7_embedding_lookup_20097.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource-
)model_deep_matmul_readvariableop_resource.
*model_deep_biasadd_readvariableop_resource2
.model_wide_deep_matmul_readvariableop_resource3
/model_wide_deep_biasadd_readvariableop_resource
identityѕњ
 model/embedding/embedding_lookupResourceGather&model_embedding_embedding_lookup_20062workclass_inp*
Tindices0*9
_class/
-+loc:@model/embedding/embedding_lookup/20062*+
_output_shapes
:         *
dtype02"
 model/embedding/embedding_lookup■
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0*
T0*9
_class/
-+loc:@model/embedding/embedding_lookup/20062*+
_output_shapes
:         2+
)model/embedding/embedding_lookup/Identityл
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2-
+model/embedding/embedding_lookup/Identity_1џ
"model/embedding_1/embedding_lookupResourceGather(model_embedding_1_embedding_lookup_20067education_inp*
Tindices0*;
_class1
/-loc:@model/embedding_1/embedding_lookup/20067*+
_output_shapes
:         *
dtype02$
"model/embedding_1/embedding_lookupє
+model/embedding_1/embedding_lookup/IdentityIdentity+model/embedding_1/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_1/embedding_lookup/20067*+
_output_shapes
:         2-
+model/embedding_1/embedding_lookup/Identityо
-model/embedding_1/embedding_lookup/Identity_1Identity4model/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2/
-model/embedding_1/embedding_lookup/Identity_1Ъ
"model/embedding_2/embedding_lookupResourceGather(model_embedding_2_embedding_lookup_20072marital_status_inp*
Tindices0*;
_class1
/-loc:@model/embedding_2/embedding_lookup/20072*+
_output_shapes
:         *
dtype02$
"model/embedding_2/embedding_lookupє
+model/embedding_2/embedding_lookup/IdentityIdentity+model/embedding_2/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_2/embedding_lookup/20072*+
_output_shapes
:         2-
+model/embedding_2/embedding_lookup/Identityо
-model/embedding_2/embedding_lookup/Identity_1Identity4model/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2/
-model/embedding_2/embedding_lookup/Identity_1Џ
"model/embedding_3/embedding_lookupResourceGather(model_embedding_3_embedding_lookup_20077occupation_inp*
Tindices0*;
_class1
/-loc:@model/embedding_3/embedding_lookup/20077*+
_output_shapes
:         *
dtype02$
"model/embedding_3/embedding_lookupє
+model/embedding_3/embedding_lookup/IdentityIdentity+model/embedding_3/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_3/embedding_lookup/20077*+
_output_shapes
:         2-
+model/embedding_3/embedding_lookup/Identityо
-model/embedding_3/embedding_lookup/Identity_1Identity4model/embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2/
-model/embedding_3/embedding_lookup/Identity_1Ю
"model/embedding_4/embedding_lookupResourceGather(model_embedding_4_embedding_lookup_20082relationship_inp*
Tindices0*;
_class1
/-loc:@model/embedding_4/embedding_lookup/20082*+
_output_shapes
:         *
dtype02$
"model/embedding_4/embedding_lookupє
+model/embedding_4/embedding_lookup/IdentityIdentity+model/embedding_4/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_4/embedding_lookup/20082*+
_output_shapes
:         2-
+model/embedding_4/embedding_lookup/Identityо
-model/embedding_4/embedding_lookup/Identity_1Identity4model/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2/
-model/embedding_4/embedding_lookup/Identity_1Ћ
"model/embedding_5/embedding_lookupResourceGather(model_embedding_5_embedding_lookup_20087race_inp*
Tindices0*;
_class1
/-loc:@model/embedding_5/embedding_lookup/20087*+
_output_shapes
:         *
dtype02$
"model/embedding_5/embedding_lookupє
+model/embedding_5/embedding_lookup/IdentityIdentity+model/embedding_5/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_5/embedding_lookup/20087*+
_output_shapes
:         2-
+model/embedding_5/embedding_lookup/Identityо
-model/embedding_5/embedding_lookup/Identity_1Identity4model/embedding_5/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2/
-model/embedding_5/embedding_lookup/Identity_1Ќ
"model/embedding_6/embedding_lookupResourceGather(model_embedding_6_embedding_lookup_20092
gender_inp*
Tindices0*;
_class1
/-loc:@model/embedding_6/embedding_lookup/20092*+
_output_shapes
:         *
dtype02$
"model/embedding_6/embedding_lookupє
+model/embedding_6/embedding_lookup/IdentityIdentity+model/embedding_6/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_6/embedding_lookup/20092*+
_output_shapes
:         2-
+model/embedding_6/embedding_lookup/Identityо
-model/embedding_6/embedding_lookup/Identity_1Identity4model/embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2/
-model/embedding_6/embedding_lookup/Identity_1Ъ
"model/embedding_7/embedding_lookupResourceGather(model_embedding_7_embedding_lookup_20097native_country_inp*
Tindices0*;
_class1
/-loc:@model/embedding_7/embedding_lookup/20097*+
_output_shapes
:         *
dtype02$
"model/embedding_7/embedding_lookupє
+model/embedding_7/embedding_lookup/IdentityIdentity+model/embedding_7/embedding_lookup:output:0*
T0*;
_class1
/-loc:@model/embedding_7/embedding_lookup/20097*+
_output_shapes
:         2-
+model/embedding_7/embedding_lookup/Identityо
-model/embedding_7/embedding_lookup/Identity_1Identity4model/embedding_7/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2/
-model/embedding_7/embedding_lookup/Identity_1`
model/reshape/ShapeShapeage_in*
T0*
_output_shapes
:2
model/reshape/Shapeљ
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/reshape/strided_slice/stackћ
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_1ћ
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_2Х
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape/strided_sliceђ
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/1ђ
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/2Т
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/reshape/Reshape/shapeЮ
model/reshape/ReshapeReshapeage_in$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
model/reshape/Reshapem
model/reshape_1/ShapeShapecapital_gain_in*
T0*
_output_shapes
:2
model/reshape_1/Shapeћ
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/reshape_1/strided_slice/stackў
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_1/strided_slice/stack_1ў
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_1/strided_slice/stack_2┬
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape_1/strided_sliceё
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_1/Reshape/shape/1ё
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_1/Reshape/shape/2­
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/reshape_1/Reshape/shapeг
model/reshape_1/ReshapeReshapecapital_gain_in&model/reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
model/reshape_1/Reshapem
model/reshape_2/ShapeShapecapital_loss_in*
T0*
_output_shapes
:2
model/reshape_2/Shapeћ
#model/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/reshape_2/strided_slice/stackў
%model/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_2/strided_slice/stack_1ў
%model/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_2/strided_slice/stack_2┬
model/reshape_2/strided_sliceStridedSlicemodel/reshape_2/Shape:output:0,model/reshape_2/strided_slice/stack:output:0.model/reshape_2/strided_slice/stack_1:output:0.model/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape_2/strided_sliceё
model/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_2/Reshape/shape/1ё
model/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_2/Reshape/shape/2­
model/reshape_2/Reshape/shapePack&model/reshape_2/strided_slice:output:0(model/reshape_2/Reshape/shape/1:output:0(model/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/reshape_2/Reshape/shapeг
model/reshape_2/ReshapeReshapecapital_loss_in&model/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
model/reshape_2/Reshapeo
model/reshape_3/ShapeShapehours_per_week_in*
T0*
_output_shapes
:2
model/reshape_3/Shapeћ
#model/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/reshape_3/strided_slice/stackў
%model/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_3/strided_slice/stack_1ў
%model/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_3/strided_slice/stack_2┬
model/reshape_3/strided_sliceStridedSlicemodel/reshape_3/Shape:output:0,model/reshape_3/strided_slice/stack:output:0.model/reshape_3/strided_slice/stack_1:output:0.model/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape_3/strided_sliceё
model/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_3/Reshape/shape/1ё
model/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_3/Reshape/shape/2­
model/reshape_3/Reshape/shapePack&model/reshape_3/strided_slice:output:0(model/reshape_3/Reshape/shape/1:output:0(model/reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/reshape_3/Reshape/shape«
model/reshape_3/ReshapeReshapehours_per_week_in&model/reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
model/reshape_3/Reshapeђ
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axisв
model/concatenate/concatConcatV24model/embedding/embedding_lookup/Identity_1:output:06model/embedding_1/embedding_lookup/Identity_1:output:06model/embedding_2/embedding_lookup/Identity_1:output:06model/embedding_3/embedding_lookup/Identity_1:output:06model/embedding_4/embedding_lookup/Identity_1:output:06model/embedding_5/embedding_lookup/Identity_1:output:06model/embedding_6/embedding_lookup/Identity_1:output:06model/embedding_7/embedding_lookup/Identity_1:output:0model/reshape/Reshape:output:0 model/reshape_1/Reshape:output:0 model/reshape_2/Reshape:output:0 model/reshape_3/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:         D2
model/concatenate/concat{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    D   2
model/flatten/Constг
model/flatten/ReshapeReshape!model/concatenate/concat:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:         D2
model/flatten/Reshape▒
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:D2*
dtype02#
!model/dense/MatMul/ReadVariableOp»
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
model/dense/MatMul░
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02$
"model/dense/BiasAdd/ReadVariableOp▒
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         22
model/dense/Reluј
model/dropout/IdentityIdentitymodel/dense/Relu:activations:0*
T0*'
_output_shapes
:         22
model/dropout/Identity«
 model/deep/MatMul/ReadVariableOpReadVariableOp)model_deep_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02"
 model/deep/MatMul/ReadVariableOpГ
model/deep/MatMulMatMulmodel/dropout/Identity:output:0(model/deep/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/deep/MatMulГ
!model/deep/BiasAdd/ReadVariableOpReadVariableOp*model_deep_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!model/deep/BiasAdd/ReadVariableOpГ
model/deep/BiasAddBiasAddmodel/deep/MatMul:product:0)model/deep/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/deep/BiasAddy
model/deep/ReluRelumodel/deep/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/deep/ReluЉ
model/dropout_1/IdentityIdentitymodel/deep/Relu:activations:0*
T0*'
_output_shapes
:         2
model/dropout_1/Identityё
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
model/concatenate_1/concat/axisМ
model/concatenate_1/concatConcatV2wide!model/dropout_1/Identity:output:0(model/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ј2
model/concatenate_1/concatЙ
%model/wide_deep/MatMul/ReadVariableOpReadVariableOp.model_wide_deep_matmul_readvariableop_resource*
_output_shapes
:	ј*
dtype02'
%model/wide_deep/MatMul/ReadVariableOp└
model/wide_deep/MatMulMatMul#model/concatenate_1/concat:output:0-model/wide_deep/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/wide_deep/MatMul╝
&model/wide_deep/BiasAdd/ReadVariableOpReadVariableOp/model_wide_deep_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model/wide_deep/BiasAdd/ReadVariableOp┴
model/wide_deep/BiasAddBiasAdd model/wide_deep/MatMul:product:0.model/wide_deep/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/wide_deep/BiasAddЉ
model/wide_deep/SigmoidSigmoid model/wide_deep/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model/wide_deep/Sigmoido
IdentityIdentitymodel/wide_deep/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         :::::::::::::::N J
(
_output_shapes
:         Щ

_user_specified_namewide:VR
'
_output_shapes
:         
'
_user_specified_nameworkclass_inp:VR
'
_output_shapes
:         
'
_user_specified_nameeducation_inp:[W
'
_output_shapes
:         
,
_user_specified_namemarital_status_inp:WS
'
_output_shapes
:         
(
_user_specified_nameoccupation_inp:YU
'
_output_shapes
:         
*
_user_specified_namerelationship_inp:QM
'
_output_shapes
:         
"
_user_specified_name
race_inp:SO
'
_output_shapes
:         
$
_user_specified_name
gender_inp:[W
'
_output_shapes
:         
,
_user_specified_namenative_country_inp:O	K
'
_output_shapes
:         
 
_user_specified_nameage_in:X
T
'
_output_shapes
:         
)
_user_specified_namecapital_gain_in:XT
'
_output_shapes
:         
)
_user_specified_namecapital_loss_in:ZV
'
_output_shapes
:         
+
_user_specified_namehours_per_week_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_3_layer_call_and_return_conditional_losses_22081

inputs
embedding_lookup_22067
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_22067inputs*
Tindices0*)
_class
loc:@embedding_lookup/22067*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22067*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22067*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_5_layer_call_and_return_conditional_losses_22145

inputs
embedding_lookup_22131
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_22131inputs*
Tindices0*)
_class
loc:@embedding_lookup/22131*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22131*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22131*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
─
q
+__inference_embedding_3_layer_call_fn_22088

inputs
unknown
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_202882
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
Л

D__inference_embedding_layer_call_and_return_conditional_losses_20201

inputs
embedding_lookup_20187
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20187inputs*
Tindices0*)
_class
loc:@embedding_lookup/20187*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20187*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1К
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20187*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
┴
t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_22463
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisѓ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         ј2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         ј2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         Щ:         :R N
(
_output_shapes
:         Щ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
╣
r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_20686

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisђ
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         ј2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         ј2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         Щ:         :P L
(
_output_shapes
:         Щ
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
░
^
B__inference_flatten_layer_call_and_return_conditional_losses_22327

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    D   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         D2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         D2

Identity"
identityIdentity:output:0**
_input_shapes
:         D:S O
+
_output_shapes
:         D
 
_user_specified_nameinputs
Ч
b
)__inference_dropout_1_layer_call_fn_22451

inputs
identityѕбStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_206612
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ЈЂ
¤
@__inference_model_layer_call_and_return_conditional_losses_21870
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12$
 embedding_embedding_lookup_21685&
"embedding_1_embedding_lookup_21690&
"embedding_2_embedding_lookup_21695&
"embedding_3_embedding_lookup_21700&
"embedding_4_embedding_lookup_21705&
"embedding_5_embedding_lookup_21710&
"embedding_6_embedding_lookup_21715&
"embedding_7_embedding_lookup_21720(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource'
#deep_matmul_readvariableop_resource(
$deep_biasadd_readvariableop_resource,
(wide_deep_matmul_readvariableop_resource-
)wide_deep_biasadd_readvariableop_resource
identityѕш
embedding/embedding_lookupResourceGather embedding_embedding_lookup_21685inputs_1*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/21685*+
_output_shapes
:         *
dtype02
embedding/embedding_lookupТ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/21685*+
_output_shapes
:         2%
#embedding/embedding_lookup/IdentityЙ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2'
%embedding/embedding_lookup/Identity_1§
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_21690inputs_2*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/21690*+
_output_shapes
:         *
dtype02
embedding_1/embedding_lookupЬ
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/21690*+
_output_shapes
:         2'
%embedding_1/embedding_lookup/Identity─
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_1/embedding_lookup/Identity_1§
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_21695inputs_3*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/21695*+
_output_shapes
:         *
dtype02
embedding_2/embedding_lookupЬ
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/21695*+
_output_shapes
:         2'
%embedding_2/embedding_lookup/Identity─
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_2/embedding_lookup/Identity_1§
embedding_3/embedding_lookupResourceGather"embedding_3_embedding_lookup_21700inputs_4*
Tindices0*5
_class+
)'loc:@embedding_3/embedding_lookup/21700*+
_output_shapes
:         *
dtype02
embedding_3/embedding_lookupЬ
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_3/embedding_lookup/21700*+
_output_shapes
:         2'
%embedding_3/embedding_lookup/Identity─
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_3/embedding_lookup/Identity_1§
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_21705inputs_5*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/21705*+
_output_shapes
:         *
dtype02
embedding_4/embedding_lookupЬ
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/21705*+
_output_shapes
:         2'
%embedding_4/embedding_lookup/Identity─
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_4/embedding_lookup/Identity_1§
embedding_5/embedding_lookupResourceGather"embedding_5_embedding_lookup_21710inputs_6*
Tindices0*5
_class+
)'loc:@embedding_5/embedding_lookup/21710*+
_output_shapes
:         *
dtype02
embedding_5/embedding_lookupЬ
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_5/embedding_lookup/21710*+
_output_shapes
:         2'
%embedding_5/embedding_lookup/Identity─
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_5/embedding_lookup/Identity_1§
embedding_6/embedding_lookupResourceGather"embedding_6_embedding_lookup_21715inputs_7*
Tindices0*5
_class+
)'loc:@embedding_6/embedding_lookup/21715*+
_output_shapes
:         *
dtype02
embedding_6/embedding_lookupЬ
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_6/embedding_lookup/21715*+
_output_shapes
:         2'
%embedding_6/embedding_lookup/Identity─
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_6/embedding_lookup/Identity_1§
embedding_7/embedding_lookupResourceGather"embedding_7_embedding_lookup_21720inputs_8*
Tindices0*5
_class+
)'loc:@embedding_7/embedding_lookup/21720*+
_output_shapes
:         *
dtype02
embedding_7/embedding_lookupЬ
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_7/embedding_lookup/21720*+
_output_shapes
:         2'
%embedding_7/embedding_lookup/Identity─
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_7/embedding_lookup/Identity_1V
reshape/ShapeShapeinputs_9*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2╚
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЇ
reshape/ReshapeReshapeinputs_9reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape/Reshape[
reshape_1/ShapeShape	inputs_10*
T0*
_output_shapes
:2
reshape_1/Shapeѕ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackї
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1ї
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2ъ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2м
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shapeћ
reshape_1/ReshapeReshape	inputs_10 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_1/Reshape[
reshape_2/ShapeShape	inputs_11*
T0*
_output_shapes
:2
reshape_2/Shapeѕ
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stackї
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1ї
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2ъ
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2м
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shapeћ
reshape_2/ReshapeReshape	inputs_11 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_2/Reshape[
reshape_3/ShapeShape	inputs_12*
T0*
_output_shapes
:2
reshape_3/Shapeѕ
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stackї
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1ї
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2ъ
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2м
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shapeћ
reshape_3/ReshapeReshape	inputs_12 reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_3/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЉ
concatenate/concatConcatV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:00embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:00embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:00embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:         D2
concatenate/concato
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    D   2
flatten/Constћ
flatten/ReshapeReshapeconcatenate/concat:output:0flatten/Const:output:0*
T0*'
_output_shapes
:         D2
flatten/ReshapeЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:D2*
dtype02
dense/MatMul/ReadVariableOpЌ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         22

dense/Relu|
dropout/IdentityIdentitydense/Relu:activations:0*
T0*'
_output_shapes
:         22
dropout/Identityю
deep/MatMul/ReadVariableOpReadVariableOp#deep_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
deep/MatMul/ReadVariableOpЋ
deep/MatMulMatMuldropout/Identity:output:0"deep/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
deep/MatMulЏ
deep/BiasAdd/ReadVariableOpReadVariableOp$deep_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
deep/BiasAdd/ReadVariableOpЋ
deep/BiasAddBiasAdddeep/MatMul:product:0#deep/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
deep/BiasAddg
	deep/ReluReludeep/BiasAdd:output:0*
T0*'
_output_shapes
:         2
	deep/Relu
dropout_1/IdentityIdentitydeep/Relu:activations:0*
T0*'
_output_shapes
:         2
dropout_1/Identityx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis┐
concatenate_1/concatConcatV2inputs_0dropout_1/Identity:output:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ј2
concatenate_1/concatг
wide_deep/MatMul/ReadVariableOpReadVariableOp(wide_deep_matmul_readvariableop_resource*
_output_shapes
:	ј*
dtype02!
wide_deep/MatMul/ReadVariableOpе
wide_deep/MatMulMatMulconcatenate_1/concat:output:0'wide_deep/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
wide_deep/MatMulф
 wide_deep/BiasAdd/ReadVariableOpReadVariableOp)wide_deep_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 wide_deep/BiasAdd/ReadVariableOpЕ
wide_deep/BiasAddBiasAddwide_deep/MatMul:product:0(wide_deep/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
wide_deep/BiasAdd
wide_deep/SigmoidSigmoidwide_deep/BiasAdd:output:0*
T0*'
_output_shapes
:         2
wide_deep/SigmoidЛ
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp embedding_embedding_lookup_21685*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addО
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_1_embedding_lookup_21690*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addО
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_2_embedding_lookup_21695*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addО
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_3_embedding_lookup_21700*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/addО
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_4_embedding_lookup_21705*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/addО
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_5_embedding_lookup_21710*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/addО
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_6_embedding_lookup_21715*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/addО
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_7_embedding_lookup_21720*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/add┐
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add┼
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1i
IdentityIdentitywide_deep/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         :::::::::::::::R N
(
_output_shapes
:         Щ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/12:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ђ
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_20661

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
юЛ
ч
@__inference_model_layer_call_and_return_conditional_losses_20945
wide
workclass_inp
education_inp
marital_status_inp
occupation_inp
relationship_inp
race_inp

gender_inp
native_country_inp

age_in
capital_gain_in
capital_loss_in
hours_per_week_in
embedding_20817
embedding_1_20820
embedding_2_20823
embedding_3_20826
embedding_4_20829
embedding_5_20832
embedding_6_20835
embedding_7_20838
dense_20847
dense_20849

deep_20853

deep_20855
wide_deep_20860
wide_deep_20862
identityѕбdeep/StatefulPartitionedCallбdense/StatefulPartitionedCallб!embedding/StatefulPartitionedCallб#embedding_1/StatefulPartitionedCallб#embedding_2/StatefulPartitionedCallб#embedding_3/StatefulPartitionedCallб#embedding_4/StatefulPartitionedCallб#embedding_5/StatefulPartitionedCallб#embedding_6/StatefulPartitionedCallб#embedding_7/StatefulPartitionedCallб!wide_deep/StatefulPartitionedCallВ
!embedding/StatefulPartitionedCallStatefulPartitionedCallworkclass_inpembedding_20817*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_202012#
!embedding/StatefulPartitionedCallЗ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCalleducation_inpembedding_1_20820*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_202302%
#embedding_1/StatefulPartitionedCallщ
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallmarital_status_inpembedding_2_20823*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_202592%
#embedding_2/StatefulPartitionedCallш
#embedding_3/StatefulPartitionedCallStatefulPartitionedCalloccupation_inpembedding_3_20826*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_202882%
#embedding_3/StatefulPartitionedCallэ
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallrelationship_inpembedding_4_20829*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_203172%
#embedding_4/StatefulPartitionedCall№
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallrace_inpembedding_5_20832*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_203462%
#embedding_5/StatefulPartitionedCallы
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall
gender_inpembedding_6_20835*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_203752%
#embedding_6/StatefulPartitionedCallщ
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallnative_country_inpembedding_7_20838*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_204042%
#embedding_7/StatefulPartitionedCall▓
reshape/PartitionedCallPartitionedCallage_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204292
reshape/PartitionedCall┴
reshape_1/PartitionedCallPartitionedCallcapital_gain_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_204502
reshape_1/PartitionedCall┴
reshape_2/PartitionedCallPartitionedCallcapital_loss_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_204712
reshape_2/PartitionedCall├
reshape_3/PartitionedCallPartitionedCallhours_per_week_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204922
reshape_3/PartitionedCallй
concatenate/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0,embedding_5/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0,embedding_7/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_205172
concatenate/PartitionedCall╠
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_205422
flatten/PartitionedCallЩ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_20847dense_20849*
Tin
2*
Tout
2*'
_output_shapes
:         2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_205762
dense/StatefulPartitionedCall╬
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         2* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_206092
dropout/PartitionedCallш
deep/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
deep_20853
deep_20855*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_deep_layer_call_and_return_conditional_losses_206332
deep/StatefulPartitionedCallМ
dropout_1/PartitionedCallPartitionedCall%deep/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_206662
dropout_1/PartitionedCallС
concatenate_1/PartitionedCallPartitionedCallwide"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ј* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_206862
concatenate_1/PartitionedCallћ
!wide_deep/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0wide_deep_20860wide_deep_20862*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_wide_deep_layer_call_and_return_conditional_losses_207062#
!wide_deep/StatefulPartitionedCall└
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_20817*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addк
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_20820*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addк
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_20823*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addк
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_20826*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/addк
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_20829*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/addк
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_5_20832*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/addк
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_6_20835*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/addк
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7_20838*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/addд
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_20847*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addг
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_20847*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1Ј
IdentityIdentity*wide_deep/StatefulPartitionedCall:output:0^deep/StatefulPartitionedCall^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall"^wide_deep/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::2<
deep/StatefulPartitionedCalldeep/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2F
!wide_deep/StatefulPartitionedCall!wide_deep/StatefulPartitionedCall:N J
(
_output_shapes
:         Щ

_user_specified_namewide:VR
'
_output_shapes
:         
'
_user_specified_nameworkclass_inp:VR
'
_output_shapes
:         
'
_user_specified_nameeducation_inp:[W
'
_output_shapes
:         
,
_user_specified_namemarital_status_inp:WS
'
_output_shapes
:         
(
_user_specified_nameoccupation_inp:YU
'
_output_shapes
:         
*
_user_specified_namerelationship_inp:QM
'
_output_shapes
:         
"
_user_specified_name
race_inp:SO
'
_output_shapes
:         
$
_user_specified_name
gender_inp:[W
'
_output_shapes
:         
,
_user_specified_namenative_country_inp:O	K
'
_output_shapes
:         
 
_user_specified_nameage_in:X
T
'
_output_shapes
:         
)
_user_specified_namecapital_gain_in:XT
'
_output_shapes
:         
)
_user_specified_namecapital_loss_in:ZV
'
_output_shapes
:         
+
_user_specified_namehours_per_week_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■

a
B__inference_dropout_layer_call_and_return_conditional_losses_20604

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
ч
u
__inference_loss_fn_3_22541E
Aembedding_3_embeddings_regularizer_square_readvariableop_resource
identityѕШ
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_3_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/addm
IdentityIdentity*embedding_3/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_5_layer_call_and_return_conditional_losses_20346

inputs
embedding_lookup_20332
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20332inputs*
Tindices0*)
_class
loc:@embedding_lookup/20332*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20332*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20332*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_4_layer_call_and_return_conditional_losses_20317

inputs
embedding_lookup_20303
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20303inputs*
Tindices0*)
_class
loc:@embedding_lookup/20303*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20303*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20303*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_2_layer_call_and_return_conditional_losses_20259

inputs
embedding_lookup_20245
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20245inputs*
Tindices0*)
_class
loc:@embedding_lookup/20245*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20245*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20245*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
Ў
ђ
F__inference_concatenate_layer_call_and_return_conditional_losses_22305
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisв
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*+
_output_shapes
:         D2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:         D2

Identity"
identityIdentity:output:0*Е
_input_shapesЌ
ћ:         :         :         :         :         :         :         :         :         :         :         :         :U Q
+
_output_shapes
:         
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/3:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/4:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/5:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/6:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/7:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/8:U	Q
+
_output_shapes
:         
"
_user_specified_name
inputs/9:V
R
+
_output_shapes
:         
#
_user_specified_name	inputs/10:VR
+
_output_shapes
:         
#
_user_specified_name	inputs/11
─
q
+__inference_embedding_6_layer_call_fn_22184

inputs
unknown
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_203752
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ч
■
F__inference_concatenate_layer_call_and_return_conditional_losses_20517

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisж
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11concat/axis:output:0*
N*
T0*+
_output_shapes
:         D2
concatg
IdentityIdentityconcat:output:0*
T0*+
_output_shapes
:         D2

Identity"
identityIdentity:output:0*Е
_input_shapesЌ
ћ:         :         :         :         :         :         :         :         :         :         :         :         :S O
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs:S	O
+
_output_shapes
:         
 
_user_specified_nameinputs:S
O
+
_output_shapes
:         
 
_user_specified_nameinputs:SO
+
_output_shapes
:         
 
_user_specified_nameinputs
Ь
z
%__inference_dense_layer_call_fn_22382

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall╬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_205762
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         D::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         D
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┼
`
B__inference_dropout_layer_call_and_return_conditional_losses_22399

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
ч
u
__inference_loss_fn_6_22580E
Aembedding_6_embeddings_regularizer_square_readvariableop_resource
identityѕШ
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_6_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/addm
IdentityIdentity*embedding_6/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
ђ
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_22441

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
К¤
й
@__inference_model_layer_call_and_return_conditional_losses_21291

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
embedding_21163
embedding_1_21166
embedding_2_21169
embedding_3_21172
embedding_4_21175
embedding_5_21178
embedding_6_21181
embedding_7_21184
dense_21193
dense_21195

deep_21199

deep_21201
wide_deep_21206
wide_deep_21208
identityѕбdeep/StatefulPartitionedCallбdense/StatefulPartitionedCallб!embedding/StatefulPartitionedCallб#embedding_1/StatefulPartitionedCallб#embedding_2/StatefulPartitionedCallб#embedding_3/StatefulPartitionedCallб#embedding_4/StatefulPartitionedCallб#embedding_5/StatefulPartitionedCallб#embedding_6/StatefulPartitionedCallб#embedding_7/StatefulPartitionedCallб!wide_deep/StatefulPartitionedCallу
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_21163*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_202012#
!embedding/StatefulPartitionedCall№
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_1_21166*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_202302%
#embedding_1/StatefulPartitionedCall№
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_2_21169*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_202592%
#embedding_2/StatefulPartitionedCall№
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs_4embedding_3_21172*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_202882%
#embedding_3/StatefulPartitionedCall№
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs_5embedding_4_21175*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_203172%
#embedding_4/StatefulPartitionedCall№
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs_6embedding_5_21178*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_203462%
#embedding_5/StatefulPartitionedCall№
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinputs_7embedding_6_21181*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_203752%
#embedding_6/StatefulPartitionedCall№
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputs_8embedding_7_21184*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_204042%
#embedding_7/StatefulPartitionedCall┤
reshape/PartitionedCallPartitionedCallinputs_9*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204292
reshape/PartitionedCall╗
reshape_1/PartitionedCallPartitionedCall	inputs_10*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_204502
reshape_1/PartitionedCall╗
reshape_2/PartitionedCallPartitionedCall	inputs_11*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_204712
reshape_2/PartitionedCall╗
reshape_3/PartitionedCallPartitionedCall	inputs_12*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204922
reshape_3/PartitionedCallй
concatenate/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0,embedding_5/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0,embedding_7/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_205172
concatenate/PartitionedCall╠
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_205422
flatten/PartitionedCallЩ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_21193dense_21195*
Tin
2*
Tout
2*'
_output_shapes
:         2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_205762
dense/StatefulPartitionedCall╬
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         2* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_206092
dropout/PartitionedCallш
deep/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
deep_21199
deep_21201*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_deep_layer_call_and_return_conditional_losses_206332
deep/StatefulPartitionedCallМ
dropout_1/PartitionedCallPartitionedCall%deep/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_206662
dropout_1/PartitionedCallТ
concatenate_1/PartitionedCallPartitionedCallinputs"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ј* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_206862
concatenate_1/PartitionedCallћ
!wide_deep/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0wide_deep_21206wide_deep_21208*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_wide_deep_layer_call_and_return_conditional_losses_207062#
!wide_deep/StatefulPartitionedCall└
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_21163*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addк
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_21166*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addк
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_21169*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addк
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_21172*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/addк
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_21175*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/addк
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_5_21178*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/addк
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_6_21181*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/addк
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7_21184*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/addд
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_21193*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addг
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21193*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1Ј
IdentityIdentity*wide_deep/StatefulPartitionedCall:output:0^deep/StatefulPartitionedCall^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall"^wide_deep/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::2<
deep/StatefulPartitionedCalldeep/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2F
!wide_deep/StatefulPartitionedCall!wide_deep/StatefulPartitionedCall:P L
(
_output_shapes
:         Щ
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:O	K
'
_output_shapes
:         
 
_user_specified_nameinputs:O
K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
q
+__inference_embedding_4_layer_call_fn_22120

inputs
unknown
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_203172
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_4_layer_call_and_return_conditional_losses_22113

inputs
embedding_lookup_22099
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_22099inputs*
Tindices0*)
_class
loc:@embedding_lookup/22099*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22099*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22099*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
█
^
B__inference_reshape_layer_call_and_return_conditional_losses_20429

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
q
+__inference_embedding_7_layer_call_fn_22216

inputs
unknown
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_204042
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
П
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_22283

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Я
Д
?__inference_deep_layer_call_and_return_conditional_losses_20633

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2:::O K
'
_output_shapes
:         2
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
П
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_20450

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┴м
Ѓ
@__inference_model_layer_call_and_return_conditional_losses_21103

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
embedding_20975
embedding_1_20978
embedding_2_20981
embedding_3_20984
embedding_4_20987
embedding_5_20990
embedding_6_20993
embedding_7_20996
dense_21005
dense_21007

deep_21011

deep_21013
wide_deep_21018
wide_deep_21020
identityѕбdeep/StatefulPartitionedCallбdense/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!embedding/StatefulPartitionedCallб#embedding_1/StatefulPartitionedCallб#embedding_2/StatefulPartitionedCallб#embedding_3/StatefulPartitionedCallб#embedding_4/StatefulPartitionedCallб#embedding_5/StatefulPartitionedCallб#embedding_6/StatefulPartitionedCallб#embedding_7/StatefulPartitionedCallб!wide_deep/StatefulPartitionedCallу
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_20975*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_202012#
!embedding/StatefulPartitionedCall№
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_1_20978*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_202302%
#embedding_1/StatefulPartitionedCall№
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_2_20981*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_202592%
#embedding_2/StatefulPartitionedCall№
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs_4embedding_3_20984*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_202882%
#embedding_3/StatefulPartitionedCall№
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs_5embedding_4_20987*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_203172%
#embedding_4/StatefulPartitionedCall№
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs_6embedding_5_20990*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_203462%
#embedding_5/StatefulPartitionedCall№
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinputs_7embedding_6_20993*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_203752%
#embedding_6/StatefulPartitionedCall№
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallinputs_8embedding_7_20996*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_204042%
#embedding_7/StatefulPartitionedCall┤
reshape/PartitionedCallPartitionedCallinputs_9*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204292
reshape/PartitionedCall╗
reshape_1/PartitionedCallPartitionedCall	inputs_10*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_204502
reshape_1/PartitionedCall╗
reshape_2/PartitionedCallPartitionedCall	inputs_11*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_204712
reshape_2/PartitionedCall╗
reshape_3/PartitionedCallPartitionedCall	inputs_12*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204922
reshape_3/PartitionedCallй
concatenate/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0,embedding_5/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0,embedding_7/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_205172
concatenate/PartitionedCall╠
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_205422
flatten/PartitionedCallЩ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_21005dense_21007*
Tin
2*
Tout
2*'
_output_shapes
:         2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_205762
dense/StatefulPartitionedCallТ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         2* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_206042!
dropout/StatefulPartitionedCall§
deep/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
deep_21011
deep_21013*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_deep_layer_call_and_return_conditional_losses_206332
deep/StatefulPartitionedCallЇ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%deep/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_206612#
!dropout_1/StatefulPartitionedCallЬ
concatenate_1/PartitionedCallPartitionedCallinputs*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ј* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_206862
concatenate_1/PartitionedCallћ
!wide_deep/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0wide_deep_21018wide_deep_21020*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_wide_deep_layer_call_and_return_conditional_losses_207062#
!wide_deep/StatefulPartitionedCall└
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_20975*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addк
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_20978*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addк
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_20981*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addк
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_20984*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/addк
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_20987*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/addк
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_5_20990*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/addк
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_6_20993*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/addк
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7_20996*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/addд
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_21005*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addг
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_21005*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1Н
IdentityIdentity*wide_deep/StatefulPartitionedCall:output:0^deep/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall"^wide_deep/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::2<
deep/StatefulPartitionedCalldeep/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2F
!wide_deep/StatefulPartitionedCall!wide_deep/StatefulPartitionedCall:P L
(
_output_shapes
:         Щ
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:O	K
'
_output_shapes
:         
 
_user_specified_nameinputs:O
K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
q
+__inference_embedding_2_layer_call_fn_22056

inputs
unknown
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_202592
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
Ёя
М
!__inference__traced_restore_22970
file_prefix)
%assignvariableop_embedding_embeddings-
)assignvariableop_1_embedding_1_embeddings-
)assignvariableop_2_embedding_2_embeddings-
)assignvariableop_3_embedding_3_embeddings-
)assignvariableop_4_embedding_4_embeddings-
)assignvariableop_5_embedding_5_embeddings-
)assignvariableop_6_embedding_6_embeddings-
)assignvariableop_7_embedding_7_embeddings#
assignvariableop_8_dense_kernel!
assignvariableop_9_dense_bias#
assignvariableop_10_deep_kernel!
assignvariableop_11_deep_bias(
$assignvariableop_12_wide_deep_kernel&
"assignvariableop_13_wide_deep_bias!
assignvariableop_14_adam_iter#
assignvariableop_15_adam_beta_1#
assignvariableop_16_adam_beta_2"
assignvariableop_17_adam_decay*
&assignvariableop_18_adam_learning_rate
assignvariableop_19_total
assignvariableop_20_count
assignvariableop_21_total_1
assignvariableop_22_count_13
/assignvariableop_23_adam_embedding_embeddings_m5
1assignvariableop_24_adam_embedding_1_embeddings_m5
1assignvariableop_25_adam_embedding_2_embeddings_m5
1assignvariableop_26_adam_embedding_3_embeddings_m5
1assignvariableop_27_adam_embedding_4_embeddings_m5
1assignvariableop_28_adam_embedding_5_embeddings_m5
1assignvariableop_29_adam_embedding_6_embeddings_m5
1assignvariableop_30_adam_embedding_7_embeddings_m+
'assignvariableop_31_adam_dense_kernel_m)
%assignvariableop_32_adam_dense_bias_m*
&assignvariableop_33_adam_deep_kernel_m(
$assignvariableop_34_adam_deep_bias_m/
+assignvariableop_35_adam_wide_deep_kernel_m-
)assignvariableop_36_adam_wide_deep_bias_m3
/assignvariableop_37_adam_embedding_embeddings_v5
1assignvariableop_38_adam_embedding_1_embeddings_v5
1assignvariableop_39_adam_embedding_2_embeddings_v5
1assignvariableop_40_adam_embedding_3_embeddings_v5
1assignvariableop_41_adam_embedding_4_embeddings_v5
1assignvariableop_42_adam_embedding_5_embeddings_v5
1assignvariableop_43_adam_embedding_6_embeddings_v5
1assignvariableop_44_adam_embedding_7_embeddings_v+
'assignvariableop_45_adam_dense_kernel_v)
%assignvariableop_46_adam_dense_bias_v*
&assignvariableop_47_adam_deep_kernel_v(
$assignvariableop_48_adam_deep_bias_v/
+assignvariableop_49_adam_wide_deep_kernel_v-
)assignvariableop_50_adam_wide_deep_bias_v
identity_52ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1┌
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*Т
value▄B┘3B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЗ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesГ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Р
_output_shapes¤
╠:::::::::::::::::::::::::::::::::::::::::::::::::::*A
dtypes7
523	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЋ
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ъ
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ъ
AssignVariableOp_2AssignVariableOp)assignvariableop_2_embedding_2_embeddingsIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ъ
AssignVariableOp_3AssignVariableOp)assignvariableop_3_embedding_3_embeddingsIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ъ
AssignVariableOp_4AssignVariableOp)assignvariableop_4_embedding_4_embeddingsIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ъ
AssignVariableOp_5AssignVariableOp)assignvariableop_5_embedding_5_embeddingsIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ъ
AssignVariableOp_6AssignVariableOp)assignvariableop_6_embedding_6_embeddingsIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ъ
AssignVariableOp_7AssignVariableOp)assignvariableop_7_embedding_7_embeddingsIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ћ
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Њ
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10ў
AssignVariableOp_10AssignVariableOpassignvariableop_10_deep_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11ќ
AssignVariableOp_11AssignVariableOpassignvariableop_11_deep_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ю
AssignVariableOp_12AssignVariableOp$assignvariableop_12_wide_deep_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Џ
AssignVariableOp_13AssignVariableOp"assignvariableop_13_wide_deep_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0	*
_output_shapes
:2
Identity_14ќ
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15ў
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16ў
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Ќ
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ъ
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19њ
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20њ
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21ћ
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22ћ
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23е
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24ф
AssignVariableOp_24AssignVariableOp1assignvariableop_24_adam_embedding_1_embeddings_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25ф
AssignVariableOp_25AssignVariableOp1assignvariableop_25_adam_embedding_2_embeddings_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26ф
AssignVariableOp_26AssignVariableOp1assignvariableop_26_adam_embedding_3_embeddings_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27ф
AssignVariableOp_27AssignVariableOp1assignvariableop_27_adam_embedding_4_embeddings_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28ф
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_embedding_5_embeddings_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29ф
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_embedding_6_embeddings_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30ф
AssignVariableOp_30AssignVariableOp1assignvariableop_30_adam_embedding_7_embeddings_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31а
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32ъ
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ъ
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_deep_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Ю
AssignVariableOp_34AssignVariableOp$assignvariableop_34_adam_deep_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35ц
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_wide_deep_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36б
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_wide_deep_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37е
AssignVariableOp_37AssignVariableOp/assignvariableop_37_adam_embedding_embeddings_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38ф
AssignVariableOp_38AssignVariableOp1assignvariableop_38_adam_embedding_1_embeddings_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39ф
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_embedding_2_embeddings_vIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40ф
AssignVariableOp_40AssignVariableOp1assignvariableop_40_adam_embedding_3_embeddings_vIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41ф
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_embedding_4_embeddings_vIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42ф
AssignVariableOp_42AssignVariableOp1assignvariableop_42_adam_embedding_5_embeddings_vIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43ф
AssignVariableOp_43AssignVariableOp1assignvariableop_43_adam_embedding_6_embeddings_vIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44ф
AssignVariableOp_44AssignVariableOp1assignvariableop_44_adam_embedding_7_embeddings_vIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45а
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_vIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46ъ
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_vIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47Ъ
AssignVariableOp_47AssignVariableOp&assignvariableop_47_adam_deep_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48Ю
AssignVariableOp_48AssignVariableOp$assignvariableop_48_adam_deep_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49ц
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_wide_deep_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50б
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_wide_deep_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp└	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51═	
Identity_52IdentityIdentity_51:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_52"#
identity_52Identity_52:output:0*с
_input_shapesЛ
╬: :::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: 
ч
u
__inference_loss_fn_4_22554E
Aembedding_4_embeddings_regularizer_square_readvariableop_resource
identityѕШ
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_4_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/addm
IdentityIdentity*embedding_4/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
В
C
'__inference_dropout_layer_call_fn_22409

inputs
identityъ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         2* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_206092
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
Э
`
'__inference_dropout_layer_call_fn_22404

inputs
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         2* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_206042
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*&
_input_shapes
:         222
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
В
y
$__inference_deep_layer_call_fn_22429

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_deep_layer_call_and_return_conditional_losses_206332
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Я
Д
?__inference_deep_layer_call_and_return_conditional_losses_22420

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         2:::O K
'
_output_shapes
:         2
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┤Њ
¤
@__inference_model_layer_call_and_return_conditional_losses_21670
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12$
 embedding_embedding_lookup_21471&
"embedding_1_embedding_lookup_21476&
"embedding_2_embedding_lookup_21481&
"embedding_3_embedding_lookup_21486&
"embedding_4_embedding_lookup_21491&
"embedding_5_embedding_lookup_21496&
"embedding_6_embedding_lookup_21501&
"embedding_7_embedding_lookup_21506(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource'
#deep_matmul_readvariableop_resource(
$deep_biasadd_readvariableop_resource,
(wide_deep_matmul_readvariableop_resource-
)wide_deep_biasadd_readvariableop_resource
identityѕш
embedding/embedding_lookupResourceGather embedding_embedding_lookup_21471inputs_1*
Tindices0*3
_class)
'%loc:@embedding/embedding_lookup/21471*+
_output_shapes
:         *
dtype02
embedding/embedding_lookupТ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*3
_class)
'%loc:@embedding/embedding_lookup/21471*+
_output_shapes
:         2%
#embedding/embedding_lookup/IdentityЙ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2'
%embedding/embedding_lookup/Identity_1§
embedding_1/embedding_lookupResourceGather"embedding_1_embedding_lookup_21476inputs_2*
Tindices0*5
_class+
)'loc:@embedding_1/embedding_lookup/21476*+
_output_shapes
:         *
dtype02
embedding_1/embedding_lookupЬ
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_1/embedding_lookup/21476*+
_output_shapes
:         2'
%embedding_1/embedding_lookup/Identity─
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_1/embedding_lookup/Identity_1§
embedding_2/embedding_lookupResourceGather"embedding_2_embedding_lookup_21481inputs_3*
Tindices0*5
_class+
)'loc:@embedding_2/embedding_lookup/21481*+
_output_shapes
:         *
dtype02
embedding_2/embedding_lookupЬ
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_2/embedding_lookup/21481*+
_output_shapes
:         2'
%embedding_2/embedding_lookup/Identity─
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_2/embedding_lookup/Identity_1§
embedding_3/embedding_lookupResourceGather"embedding_3_embedding_lookup_21486inputs_4*
Tindices0*5
_class+
)'loc:@embedding_3/embedding_lookup/21486*+
_output_shapes
:         *
dtype02
embedding_3/embedding_lookupЬ
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_3/embedding_lookup/21486*+
_output_shapes
:         2'
%embedding_3/embedding_lookup/Identity─
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_3/embedding_lookup/Identity_1§
embedding_4/embedding_lookupResourceGather"embedding_4_embedding_lookup_21491inputs_5*
Tindices0*5
_class+
)'loc:@embedding_4/embedding_lookup/21491*+
_output_shapes
:         *
dtype02
embedding_4/embedding_lookupЬ
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_4/embedding_lookup/21491*+
_output_shapes
:         2'
%embedding_4/embedding_lookup/Identity─
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_4/embedding_lookup/Identity_1§
embedding_5/embedding_lookupResourceGather"embedding_5_embedding_lookup_21496inputs_6*
Tindices0*5
_class+
)'loc:@embedding_5/embedding_lookup/21496*+
_output_shapes
:         *
dtype02
embedding_5/embedding_lookupЬ
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_5/embedding_lookup/21496*+
_output_shapes
:         2'
%embedding_5/embedding_lookup/Identity─
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_5/embedding_lookup/Identity_1§
embedding_6/embedding_lookupResourceGather"embedding_6_embedding_lookup_21501inputs_7*
Tindices0*5
_class+
)'loc:@embedding_6/embedding_lookup/21501*+
_output_shapes
:         *
dtype02
embedding_6/embedding_lookupЬ
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_6/embedding_lookup/21501*+
_output_shapes
:         2'
%embedding_6/embedding_lookup/Identity─
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_6/embedding_lookup/Identity_1§
embedding_7/embedding_lookupResourceGather"embedding_7_embedding_lookup_21506inputs_8*
Tindices0*5
_class+
)'loc:@embedding_7/embedding_lookup/21506*+
_output_shapes
:         *
dtype02
embedding_7/embedding_lookupЬ
%embedding_7/embedding_lookup/IdentityIdentity%embedding_7/embedding_lookup:output:0*
T0*5
_class+
)'loc:@embedding_7/embedding_lookup/21506*+
_output_shapes
:         2'
%embedding_7/embedding_lookup/Identity─
'embedding_7/embedding_lookup/Identity_1Identity.embedding_7/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2)
'embedding_7/embedding_lookup/Identity_1V
reshape/ShapeShapeinputs_9*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2╚
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЇ
reshape/ReshapeReshapeinputs_9reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape/Reshape[
reshape_1/ShapeShape	inputs_10*
T0*
_output_shapes
:2
reshape_1/Shapeѕ
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stackї
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1ї
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2ъ
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2м
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shapeћ
reshape_1/ReshapeReshape	inputs_10 reshape_1/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_1/Reshape[
reshape_2/ShapeShape	inputs_11*
T0*
_output_shapes
:2
reshape_2/Shapeѕ
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stackї
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1ї
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2ъ
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2м
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shapeћ
reshape_2/ReshapeReshape	inputs_11 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_2/Reshape[
reshape_3/ShapeShape	inputs_12*
T0*
_output_shapes
:2
reshape_3/Shapeѕ
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stackї
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1ї
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2ъ
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2м
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shapeћ
reshape_3/ReshapeReshape	inputs_12 reshape_3/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
reshape_3/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisЉ
concatenate/concatConcatV2.embedding/embedding_lookup/Identity_1:output:00embedding_1/embedding_lookup/Identity_1:output:00embedding_2/embedding_lookup/Identity_1:output:00embedding_3/embedding_lookup/Identity_1:output:00embedding_4/embedding_lookup/Identity_1:output:00embedding_5/embedding_lookup/Identity_1:output:00embedding_6/embedding_lookup/Identity_1:output:00embedding_7/embedding_lookup/Identity_1:output:0reshape/Reshape:output:0reshape_1/Reshape:output:0reshape_2/Reshape:output:0reshape_3/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*+
_output_shapes
:         D2
concatenate/concato
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    D   2
flatten/Constћ
flatten/ReshapeReshapeconcatenate/concat:output:0flatten/Const:output:0*
T0*'
_output_shapes
:         D2
flatten/ReshapeЪ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:D2*
dtype02
dense/MatMul/ReadVariableOpЌ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense/MatMulъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOpЎ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         22

dense/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstЮ
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:         22
dropout/dropout/Mulv
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape╠
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yя
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         22
dropout/dropout/GreaterEqualЌ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22
dropout/dropout/Castџ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:         22
dropout/dropout/Mul_1ю
deep/MatMul/ReadVariableOpReadVariableOp#deep_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
deep/MatMul/ReadVariableOpЋ
deep/MatMulMatMuldropout/dropout/Mul_1:z:0"deep/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
deep/MatMulЏ
deep/BiasAdd/ReadVariableOpReadVariableOp$deep_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
deep/BiasAdd/ReadVariableOpЋ
deep/BiasAddBiasAdddeep/MatMul:product:0#deep/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
deep/BiasAddg
	deep/ReluReludeep/BiasAdd:output:0*
T0*'
_output_shapes
:         2
	deep/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/Constб
dropout_1/dropout/MulMuldeep/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_1/dropout/Muly
dropout_1/dropout/ShapeShapedeep/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeм
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЅ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yТ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2 
dropout_1/dropout/GreaterEqualЮ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_1/dropout/Castб
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_1/dropout/Mul_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis┐
concatenate_1/concatConcatV2inputs_0dropout_1/dropout/Mul_1:z:0"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ј2
concatenate_1/concatг
wide_deep/MatMul/ReadVariableOpReadVariableOp(wide_deep_matmul_readvariableop_resource*
_output_shapes
:	ј*
dtype02!
wide_deep/MatMul/ReadVariableOpе
wide_deep/MatMulMatMulconcatenate_1/concat:output:0'wide_deep/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
wide_deep/MatMulф
 wide_deep/BiasAdd/ReadVariableOpReadVariableOp)wide_deep_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 wide_deep/BiasAdd/ReadVariableOpЕ
wide_deep/BiasAddBiasAddwide_deep/MatMul:product:0(wide_deep/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
wide_deep/BiasAdd
wide_deep/SigmoidSigmoidwide_deep/BiasAdd:output:0*
T0*'
_output_shapes
:         2
wide_deep/SigmoidЛ
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp embedding_embedding_lookup_21471*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addО
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_1_embedding_lookup_21476*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addО
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_2_embedding_lookup_21481*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addО
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_3_embedding_lookup_21486*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/addО
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_4_embedding_lookup_21491*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/addО
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_5_embedding_lookup_21496*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/addО
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_6_embedding_lookup_21501*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/addО
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp"embedding_7_embedding_lookup_21506*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/add┐
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add┼
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1i
IdentityIdentitywide_deep/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         :::::::::::::::R N
(
_output_shapes
:         Щ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/12:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_2_layer_call_and_return_conditional_losses_22049

inputs
embedding_lookup_22035
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_22035inputs*
Tindices0*)
_class
loc:@embedding_lookup/22035*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22035*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22035*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ч
u
__inference_loss_fn_1_22515E
Aembedding_1_embeddings_regularizer_square_readvariableop_resource
identityѕШ
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_1_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addm
IdentityIdentity*embedding_1/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
К
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_20666

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Э
E
)__inference_reshape_2_layer_call_fn_22270

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_204712
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ђ
Y
-__inference_concatenate_1_layer_call_fn_22469
inputs_0
inputs_1
identity▓
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:         ј* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_206862
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ј2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         Щ:         :R N
(
_output_shapes
:         Щ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
ч
u
__inference_loss_fn_5_22567E
Aembedding_5_embeddings_regularizer_square_readvariableop_resource
identityѕШ
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_5_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/addm
IdentityIdentity*embedding_5/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ў
џ
%__inference_model_layer_call_fn_21322
wide
workclass_inp
education_inp
marital_status_inp
occupation_inp
relationship_inp
race_inp

gender_inp
native_country_inp

age_in
capital_gain_in
capital_loss_in
hours_per_week_in
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallwideworkclass_inpeducation_inpmarital_status_inpoccupation_inprelationship_inprace_inp
gender_inpnative_country_inpage_incapital_gain_incapital_loss_inhours_per_week_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*&
Tin
2*
Tout
2*'
_output_shapes
:         *0
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_212912
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
(
_output_shapes
:         Щ

_user_specified_namewide:VR
'
_output_shapes
:         
'
_user_specified_nameworkclass_inp:VR
'
_output_shapes
:         
'
_user_specified_nameeducation_inp:[W
'
_output_shapes
:         
,
_user_specified_namemarital_status_inp:WS
'
_output_shapes
:         
(
_user_specified_nameoccupation_inp:YU
'
_output_shapes
:         
*
_user_specified_namerelationship_inp:QM
'
_output_shapes
:         
"
_user_specified_name
race_inp:SO
'
_output_shapes
:         
$
_user_specified_name
gender_inp:[W
'
_output_shapes
:         
,
_user_specified_namenative_country_inp:O	K
'
_output_shapes
:         
 
_user_specified_nameage_in:X
T
'
_output_shapes
:         
)
_user_specified_namecapital_gain_in:XT
'
_output_shapes
:         
)
_user_specified_namecapital_loss_in:ZV
'
_output_shapes
:         
+
_user_specified_namehours_per_week_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_7_layer_call_and_return_conditional_losses_20404

inputs
embedding_lookup_20390
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20390inputs*
Tindices0*)
_class
loc:@embedding_lookup/20390*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20390*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20390*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
т
я
%__inference_model_layer_call_fn_21960
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*&
Tin
2*
Tout
2*'
_output_shapes
:         *0
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_212912
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         Щ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/12:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_6_layer_call_and_return_conditional_losses_20375

inputs
embedding_lookup_20361
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20361inputs*
Tindices0*)
_class
loc:@embedding_lookup/20361*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20361*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20361*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
Э
E
)__inference_reshape_1_layer_call_fn_22252

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_204502
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ѓ
Ђ
F__inference_embedding_7_layer_call_and_return_conditional_losses_22209

inputs
embedding_lookup_22195
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_22195inputs*
Tindices0*)
_class
loc:@embedding_lookup/22195*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22195*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22195*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
┼
е
@__inference_dense_layer_call_and_return_conditional_losses_20576

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Relu╣
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add┐
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1f
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         D:::O K
'
_output_shapes
:         D
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Л

D__inference_embedding_layer_call_and_return_conditional_losses_21985

inputs
embedding_lookup_21971
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_21971inputs*
Tindices0*)
_class
loc:@embedding_lookup/21971*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/21971*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1К
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_21971*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
█
^
B__inference_reshape_layer_call_and_return_conditional_losses_22229

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ў
џ
%__inference_model_layer_call_fn_21134
wide
workclass_inp
education_inp
marital_status_inp
occupation_inp
relationship_inp
race_inp

gender_inp
native_country_inp

age_in
capital_gain_in
capital_loss_in
hours_per_week_in
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallwideworkclass_inpeducation_inpmarital_status_inpoccupation_inprelationship_inprace_inp
gender_inpnative_country_inpage_incapital_gain_incapital_loss_inhours_per_week_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*&
Tin
2*
Tout
2*'
_output_shapes
:         *0
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_211032
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
(
_output_shapes
:         Щ

_user_specified_namewide:VR
'
_output_shapes
:         
'
_user_specified_nameworkclass_inp:VR
'
_output_shapes
:         
'
_user_specified_nameeducation_inp:[W
'
_output_shapes
:         
,
_user_specified_namemarital_status_inp:WS
'
_output_shapes
:         
(
_user_specified_nameoccupation_inp:YU
'
_output_shapes
:         
*
_user_specified_namerelationship_inp:QM
'
_output_shapes
:         
"
_user_specified_name
race_inp:SO
'
_output_shapes
:         
$
_user_specified_name
gender_inp:[W
'
_output_shapes
:         
,
_user_specified_namenative_country_inp:O	K
'
_output_shapes
:         
 
_user_specified_nameage_in:X
T
'
_output_shapes
:         
)
_user_specified_namecapital_gain_in:XT
'
_output_shapes
:         
)
_user_specified_namecapital_loss_in:ZV
'
_output_shapes
:         
+
_user_specified_namehours_per_week_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
К
s
__inference_loss_fn_0_22502C
?embedding_embeddings_regularizer_square_readvariableop_resource
identityѕ­
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp?embedding_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addk
IdentityIdentity(embedding/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ж
г
D__inference_wide_deep_layer_call_and_return_conditional_losses_20706

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ј:::P L
(
_output_shapes
:         ј
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
П
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_20492

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З
C
'__inference_flatten_layer_call_fn_22332

inputs
identityъ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_205422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         D2

Identity"
identityIdentity:output:0**
_input_shapes
:         D:S O
+
_output_shapes
:         D
 
_user_specified_nameinputs
э
ў
#__inference_signature_wrapper_21456

age_in
capital_gain_in
capital_loss_in
education_inp

gender_inp
hours_per_week_in
marital_status_inp
native_country_inp
occupation_inp
race_inp
relationship_inp
wide
workclass_inp
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallwideworkclass_inpeducation_inpmarital_status_inpoccupation_inprelationship_inprace_inp
gender_inpnative_country_inpage_incapital_gain_incapital_loss_inhours_per_week_inunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*&
Tin
2*
Tout
2*'
_output_shapes
:         *0
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__wrapped_model_201682
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         :         :         :         :         :         :         :         :         :         :         :         Щ:         ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameage_in:XT
'
_output_shapes
:         
)
_user_specified_namecapital_gain_in:XT
'
_output_shapes
:         
)
_user_specified_namecapital_loss_in:VR
'
_output_shapes
:         
'
_user_specified_nameeducation_inp:SO
'
_output_shapes
:         
$
_user_specified_name
gender_inp:ZV
'
_output_shapes
:         
+
_user_specified_namehours_per_week_in:[W
'
_output_shapes
:         
,
_user_specified_namemarital_status_inp:[W
'
_output_shapes
:         
,
_user_specified_namenative_country_inp:WS
'
_output_shapes
:         
(
_user_specified_nameoccupation_inp:Q	M
'
_output_shapes
:         
"
_user_specified_name
race_inp:Y
U
'
_output_shapes
:         
*
_user_specified_namerelationship_inp:NJ
(
_output_shapes
:         Щ

_user_specified_namewide:VR
'
_output_shapes
:         
'
_user_specified_nameworkclass_inp:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
т
я
%__inference_model_layer_call_fn_21915
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*&
Tin
2*
Tout
2*'
_output_shapes
:         *0
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_211032
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         Щ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/8:Q	M
'
_output_shapes
:         
"
_user_specified_name
inputs/9:R
N
'
_output_shapes
:         
#
_user_specified_name	inputs/10:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/11:RN
'
_output_shapes
:         
#
_user_specified_name	inputs/12:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
░
^
B__inference_flatten_layer_call_and_return_conditional_losses_20542

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    D   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         D2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         D2

Identity"
identityIdentity:output:0**
_input_shapes
:         D:S O
+
_output_shapes
:         D
 
_user_specified_nameinputs
ѓ
Ђ
F__inference_embedding_6_layer_call_and_return_conditional_losses_22177

inputs
embedding_lookup_22163
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_22163inputs*
Tindices0*)
_class
loc:@embedding_lookup/22163*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22163*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22163*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
┼
е
@__inference_dense_layer_call_and_return_conditional_losses_22373

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         22
Relu╣
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/add┐
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1f
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*.
_input_shapes
:         D:::O K
'
_output_shapes
:         D
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_1_layer_call_and_return_conditional_losses_22017

inputs
embedding_lookup_22003
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_22003inputs*
Tindices0*)
_class
loc:@embedding_lookup/22003*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/22003*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_22003*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
└
o
)__inference_embedding_layer_call_fn_21992

inputs
unknown
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_202012
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
Э
E
)__inference_reshape_3_layer_call_fn_22288

inputs
identityц
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204922
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
■

a
B__inference_dropout_layer_call_and_return_conditional_losses_22394

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         22
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         2*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yЙ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         22
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         22
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         22
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
─
q
+__inference_embedding_5_layer_call_fn_22152

inputs
unknown
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_203462
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
П
`
D__inference_reshape_2_layer_call_and_return_conditional_losses_22265

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
К
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_22446

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
─
q
+__inference_embedding_1_layer_call_fn_22024

inputs
unknown
identityѕбStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_202302
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ѓ
Ђ
F__inference_embedding_3_layer_call_and_return_conditional_losses_20288

inputs
embedding_lookup_20274
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20274inputs*
Tindices0*)
_class
loc:@embedding_lookup/20274*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20274*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20274*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
­
E
)__inference_dropout_1_layer_call_fn_22456

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_206662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ќн
┴
@__inference_model_layer_call_and_return_conditional_losses_20802
wide
workclass_inp
education_inp
marital_status_inp
occupation_inp
relationship_inp
race_inp

gender_inp
native_country_inp

age_in
capital_gain_in
capital_loss_in
hours_per_week_in
embedding_20210
embedding_1_20239
embedding_2_20268
embedding_3_20297
embedding_4_20326
embedding_5_20355
embedding_6_20384
embedding_7_20413
dense_20587
dense_20589

deep_20644

deep_20646
wide_deep_20717
wide_deep_20719
identityѕбdeep/StatefulPartitionedCallбdense/StatefulPartitionedCallбdropout/StatefulPartitionedCallб!dropout_1/StatefulPartitionedCallб!embedding/StatefulPartitionedCallб#embedding_1/StatefulPartitionedCallб#embedding_2/StatefulPartitionedCallб#embedding_3/StatefulPartitionedCallб#embedding_4/StatefulPartitionedCallб#embedding_5/StatefulPartitionedCallб#embedding_6/StatefulPartitionedCallб#embedding_7/StatefulPartitionedCallб!wide_deep/StatefulPartitionedCallВ
!embedding/StatefulPartitionedCallStatefulPartitionedCallworkclass_inpembedding_20210*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_202012#
!embedding/StatefulPartitionedCallЗ
#embedding_1/StatefulPartitionedCallStatefulPartitionedCalleducation_inpembedding_1_20239*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_202302%
#embedding_1/StatefulPartitionedCallщ
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallmarital_status_inpembedding_2_20268*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_2_layer_call_and_return_conditional_losses_202592%
#embedding_2/StatefulPartitionedCallш
#embedding_3/StatefulPartitionedCallStatefulPartitionedCalloccupation_inpembedding_3_20297*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_3_layer_call_and_return_conditional_losses_202882%
#embedding_3/StatefulPartitionedCallэ
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallrelationship_inpembedding_4_20326*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_4_layer_call_and_return_conditional_losses_203172%
#embedding_4/StatefulPartitionedCall№
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallrace_inpembedding_5_20355*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_5_layer_call_and_return_conditional_losses_203462%
#embedding_5/StatefulPartitionedCallы
#embedding_6/StatefulPartitionedCallStatefulPartitionedCall
gender_inpembedding_6_20384*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_6_layer_call_and_return_conditional_losses_203752%
#embedding_6/StatefulPartitionedCallщ
#embedding_7/StatefulPartitionedCallStatefulPartitionedCallnative_country_inpembedding_7_20413*
Tin
2*
Tout
2*+
_output_shapes
:         *#
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_embedding_7_layer_call_and_return_conditional_losses_204042%
#embedding_7/StatefulPartitionedCall▓
reshape/PartitionedCallPartitionedCallage_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_204292
reshape/PartitionedCall┴
reshape_1/PartitionedCallPartitionedCallcapital_gain_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_204502
reshape_1/PartitionedCall┴
reshape_2/PartitionedCallPartitionedCallcapital_loss_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_2_layer_call_and_return_conditional_losses_204712
reshape_2/PartitionedCall├
reshape_3/PartitionedCallPartitionedCallhours_per_week_in*
Tin
2*
Tout
2*+
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204922
reshape_3/PartitionedCallй
concatenate/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0,embedding_5/StatefulPartitionedCall:output:0,embedding_6/StatefulPartitionedCall:output:0,embedding_7/StatefulPartitionedCall:output:0 reshape/PartitionedCall:output:0"reshape_1/PartitionedCall:output:0"reshape_2/PartitionedCall:output:0"reshape_3/PartitionedCall:output:0*
Tin
2*
Tout
2*+
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_205172
concatenate/PartitionedCall╠
flatten/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_205422
flatten/PartitionedCallЩ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_20587dense_20589*
Tin
2*
Tout
2*'
_output_shapes
:         2*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_205762
dense/StatefulPartitionedCallТ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         2* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_206042!
dropout/StatefulPartitionedCall§
deep/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
deep_20644
deep_20646*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_deep_layer_call_and_return_conditional_losses_206332
deep/StatefulPartitionedCallЇ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%deep/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_206612#
!dropout_1/StatefulPartitionedCallВ
concatenate_1/PartitionedCallPartitionedCallwide*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ј* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_206862
concatenate_1/PartitionedCallћ
!wide_deep/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0wide_deep_20717wide_deep_20719*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_wide_deep_layer_call_and_return_conditional_losses_207062#
!wide_deep/StatefulPartitionedCall└
6embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_20210*
_output_shapes

:	*
dtype028
6embedding/embeddings/Regularizer/Square/ReadVariableOp┼
'embedding/embeddings/Regularizer/SquareSquare>embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:	2)
'embedding/embeddings/Regularizer/SquareА
&embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&embedding/embeddings/Regularizer/Constм
$embedding/embeddings/Regularizer/SumSum+embedding/embeddings/Regularizer/Square:y:0/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/SumЋ
&embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2(
&embedding/embeddings/Regularizer/mul/xн
$embedding/embeddings/Regularizer/mulMul/embedding/embeddings/Regularizer/mul/x:output:0-embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/mulЋ
&embedding/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&embedding/embeddings/Regularizer/add/xЛ
$embedding/embeddings/Regularizer/addAddV2/embedding/embeddings/Regularizer/add/x:output:0(embedding/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2&
$embedding/embeddings/Regularizer/addк
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_20239*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/addк
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_20268*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addк
8embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_20297*
_output_shapes

:*
dtype02:
8embedding_3/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_3/embeddings/Regularizer/SquareSquare@embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_3/embeddings/Regularizer/SquareЦ
(embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_3/embeddings/Regularizer/Const┌
&embedding_3/embeddings/Regularizer/SumSum-embedding_3/embeddings/Regularizer/Square:y:01embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/SumЎ
(embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_3/embeddings/Regularizer/mul/x▄
&embedding_3/embeddings/Regularizer/mulMul1embedding_3/embeddings/Regularizer/mul/x:output:0/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/mulЎ
(embedding_3/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_3/embeddings/Regularizer/add/x┘
&embedding_3/embeddings/Regularizer/addAddV21embedding_3/embeddings/Regularizer/add/x:output:0*embedding_3/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_3/embeddings/Regularizer/addк
8embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_20326*
_output_shapes

:*
dtype02:
8embedding_4/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_4/embeddings/Regularizer/SquareSquare@embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_4/embeddings/Regularizer/SquareЦ
(embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_4/embeddings/Regularizer/Const┌
&embedding_4/embeddings/Regularizer/SumSum-embedding_4/embeddings/Regularizer/Square:y:01embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/SumЎ
(embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_4/embeddings/Regularizer/mul/x▄
&embedding_4/embeddings/Regularizer/mulMul1embedding_4/embeddings/Regularizer/mul/x:output:0/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/mulЎ
(embedding_4/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_4/embeddings/Regularizer/add/x┘
&embedding_4/embeddings/Regularizer/addAddV21embedding_4/embeddings/Regularizer/add/x:output:0*embedding_4/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_4/embeddings/Regularizer/addк
8embedding_5/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_5_20355*
_output_shapes

:*
dtype02:
8embedding_5/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_5/embeddings/Regularizer/SquareSquare@embedding_5/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_5/embeddings/Regularizer/SquareЦ
(embedding_5/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_5/embeddings/Regularizer/Const┌
&embedding_5/embeddings/Regularizer/SumSum-embedding_5/embeddings/Regularizer/Square:y:01embedding_5/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/SumЎ
(embedding_5/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_5/embeddings/Regularizer/mul/x▄
&embedding_5/embeddings/Regularizer/mulMul1embedding_5/embeddings/Regularizer/mul/x:output:0/embedding_5/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/mulЎ
(embedding_5/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_5/embeddings/Regularizer/add/x┘
&embedding_5/embeddings/Regularizer/addAddV21embedding_5/embeddings/Regularizer/add/x:output:0*embedding_5/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_5/embeddings/Regularizer/addк
8embedding_6/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_6_20384*
_output_shapes

:*
dtype02:
8embedding_6/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_6/embeddings/Regularizer/SquareSquare@embedding_6/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_6/embeddings/Regularizer/SquareЦ
(embedding_6/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_6/embeddings/Regularizer/Const┌
&embedding_6/embeddings/Regularizer/SumSum-embedding_6/embeddings/Regularizer/Square:y:01embedding_6/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/SumЎ
(embedding_6/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_6/embeddings/Regularizer/mul/x▄
&embedding_6/embeddings/Regularizer/mulMul1embedding_6/embeddings/Regularizer/mul/x:output:0/embedding_6/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/mulЎ
(embedding_6/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_6/embeddings/Regularizer/add/x┘
&embedding_6/embeddings/Regularizer/addAddV21embedding_6/embeddings/Regularizer/add/x:output:0*embedding_6/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_6/embeddings/Regularizer/addк
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_7_20413*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/addд
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_20587*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addг
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_20587*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1Н
IdentityIdentity*wide_deep/StatefulPartitionedCall:output:0^deep/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall$^embedding_7/StatefulPartitionedCall"^wide_deep/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*┼
_input_shapes│
░:         Щ:         :         :         :         :         :         :         :         :         :         :         :         ::::::::::::::2<
deep/StatefulPartitionedCalldeep/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2J
#embedding_7/StatefulPartitionedCall#embedding_7/StatefulPartitionedCall2F
!wide_deep/StatefulPartitionedCall!wide_deep/StatefulPartitionedCall:N J
(
_output_shapes
:         Щ

_user_specified_namewide:VR
'
_output_shapes
:         
'
_user_specified_nameworkclass_inp:VR
'
_output_shapes
:         
'
_user_specified_nameeducation_inp:[W
'
_output_shapes
:         
,
_user_specified_namemarital_status_inp:WS
'
_output_shapes
:         
(
_user_specified_nameoccupation_inp:YU
'
_output_shapes
:         
*
_user_specified_namerelationship_inp:QM
'
_output_shapes
:         
"
_user_specified_name
race_inp:SO
'
_output_shapes
:         
$
_user_specified_name
gender_inp:[W
'
_output_shapes
:         
,
_user_specified_namenative_country_inp:O	K
'
_output_shapes
:         
 
_user_specified_nameage_in:X
T
'
_output_shapes
:         
)
_user_specified_namecapital_gain_in:XT
'
_output_shapes
:         
)
_user_specified_namecapital_loss_in:ZV
'
_output_shapes
:         
+
_user_specified_namehours_per_week_in:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч
u
__inference_loss_fn_2_22528E
Aembedding_2_embeddings_regularizer_square_readvariableop_resource
identityѕШ
8embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_2_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02:
8embedding_2/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_2/embeddings/Regularizer/SquareSquare@embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_2/embeddings/Regularizer/SquareЦ
(embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_2/embeddings/Regularizer/Const┌
&embedding_2/embeddings/Regularizer/SumSum-embedding_2/embeddings/Regularizer/Square:y:01embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/SumЎ
(embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_2/embeddings/Regularizer/mul/x▄
&embedding_2/embeddings/Regularizer/mulMul1embedding_2/embeddings/Regularizer/mul/x:output:0/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/mulЎ
(embedding_2/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_2/embeddings/Regularizer/add/x┘
&embedding_2/embeddings/Regularizer/addAddV21embedding_2/embeddings/Regularizer/add/x:output:0*embedding_2/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_2/embeddings/Regularizer/addm
IdentityIdentity*embedding_2/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
П
`
D__inference_reshape_2_layer_call_and_return_conditional_losses_20471

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
єr
Ї
__inference__traced_save_22805
file_prefix3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop5
1savev2_embedding_2_embeddings_read_readvariableop5
1savev2_embedding_3_embeddings_read_readvariableop5
1savev2_embedding_4_embeddings_read_readvariableop5
1savev2_embedding_5_embeddings_read_readvariableop5
1savev2_embedding_6_embeddings_read_readvariableop5
1savev2_embedding_7_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop*
&savev2_deep_kernel_read_readvariableop(
$savev2_deep_bias_read_readvariableop/
+savev2_wide_deep_kernel_read_readvariableop-
)savev2_wide_deep_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop<
8savev2_adam_embedding_2_embeddings_m_read_readvariableop<
8savev2_adam_embedding_3_embeddings_m_read_readvariableop<
8savev2_adam_embedding_4_embeddings_m_read_readvariableop<
8savev2_adam_embedding_5_embeddings_m_read_readvariableop<
8savev2_adam_embedding_6_embeddings_m_read_readvariableop<
8savev2_adam_embedding_7_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop1
-savev2_adam_deep_kernel_m_read_readvariableop/
+savev2_adam_deep_bias_m_read_readvariableop6
2savev2_adam_wide_deep_kernel_m_read_readvariableop4
0savev2_adam_wide_deep_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop<
8savev2_adam_embedding_2_embeddings_v_read_readvariableop<
8savev2_adam_embedding_3_embeddings_v_read_readvariableop<
8savev2_adam_embedding_4_embeddings_v_read_readvariableop<
8savev2_adam_embedding_5_embeddings_v_read_readvariableop<
8savev2_adam_embedding_6_embeddings_v_read_readvariableop<
8savev2_adam_embedding_7_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop1
-savev2_adam_deep_kernel_v_read_readvariableop/
+savev2_adam_deep_bias_v_read_readvariableop6
2savev2_adam_wide_deep_kernel_v_read_readvariableop4
0savev2_adam_wide_deep_bias_v_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ј
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5b77094237b24e85a0f8886b7e2b3974/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameн
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*Т
value▄B┘3B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-2/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-3/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-5/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-6/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-7/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesЬ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:3*
dtype0*y
valuepBn3B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesъ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop1savev2_embedding_2_embeddings_read_readvariableop1savev2_embedding_3_embeddings_read_readvariableop1savev2_embedding_4_embeddings_read_readvariableop1savev2_embedding_5_embeddings_read_readvariableop1savev2_embedding_6_embeddings_read_readvariableop1savev2_embedding_7_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop&savev2_deep_kernel_read_readvariableop$savev2_deep_bias_read_readvariableop+savev2_wide_deep_kernel_read_readvariableop)savev2_wide_deep_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop8savev2_adam_embedding_2_embeddings_m_read_readvariableop8savev2_adam_embedding_3_embeddings_m_read_readvariableop8savev2_adam_embedding_4_embeddings_m_read_readvariableop8savev2_adam_embedding_5_embeddings_m_read_readvariableop8savev2_adam_embedding_6_embeddings_m_read_readvariableop8savev2_adam_embedding_7_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop-savev2_adam_deep_kernel_m_read_readvariableop+savev2_adam_deep_bias_m_read_readvariableop2savev2_adam_wide_deep_kernel_m_read_readvariableop0savev2_adam_wide_deep_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop8savev2_adam_embedding_2_embeddings_v_read_readvariableop8savev2_adam_embedding_3_embeddings_v_read_readvariableop8savev2_adam_embedding_4_embeddings_v_read_readvariableop8savev2_adam_embedding_5_embeddings_v_read_readvariableop8savev2_adam_embedding_6_embeddings_v_read_readvariableop8savev2_adam_embedding_7_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop-savev2_adam_deep_kernel_v_read_readvariableop+savev2_adam_deep_bias_v_read_readvariableop2savev2_adam_wide_deep_kernel_v_read_readvariableop0savev2_adam_wide_deep_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *A
dtypes7
523	2
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*«
_input_shapesю
Ў: :	:::::::*:D2:2:2::	ј:: : : : : : : : : :	:::::::*:D2:2:2::	ј::	:::::::*:D2:2:2::	ј:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:*:$	 

_output_shapes

:D2: 


_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::%!

_output_shapes
:	ј: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	:$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:*:$  

_output_shapes

:D2: !

_output_shapes
:2:$" 

_output_shapes

:2: #

_output_shapes
::%$!

_output_shapes
:	ј: %

_output_shapes
::$& 

_output_shapes

:	:$' 

_output_shapes

::$( 

_output_shapes

::$) 

_output_shapes

::$* 

_output_shapes

::$+ 

_output_shapes

::$, 

_output_shapes

::$- 

_output_shapes

:*:$. 

_output_shapes

:D2: /

_output_shapes
:2:$0 

_output_shapes

:2: 1

_output_shapes
::%2!

_output_shapes
:	ј: 3

_output_shapes
::4

_output_shapes
: 
р
т
+__inference_concatenate_layer_call_fn_22321
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
identityБ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11*
Tin
2*
Tout
2*+
_output_shapes
:         D* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_205172
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         D2

Identity"
identityIdentity:output:0*Е
_input_shapesЌ
ћ:         :         :         :         :         :         :         :         :         :         :         :         :U Q
+
_output_shapes
:         
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/2:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/3:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/4:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/5:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/6:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/7:UQ
+
_output_shapes
:         
"
_user_specified_name
inputs/8:U	Q
+
_output_shapes
:         
"
_user_specified_name
inputs/9:V
R
+
_output_shapes
:         
#
_user_specified_name	inputs/10:VR
+
_output_shapes
:         
#
_user_specified_name	inputs/11
ѓ
Ђ
F__inference_embedding_1_layer_call_and_return_conditional_losses_20230

inputs
embedding_lookup_20216
identityѕ╦
embedding_lookupResourceGatherembedding_lookup_20216inputs*
Tindices0*)
_class
loc:@embedding_lookup/20216*+
_output_shapes
:         *
dtype02
embedding_lookupЙ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/20216*+
_output_shapes
:         2
embedding_lookup/Identityа
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:         2
embedding_lookup/Identity_1╦
8embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_20216*
_output_shapes

:*
dtype02:
8embedding_1/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_1/embeddings/Regularizer/SquareSquare@embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:2+
)embedding_1/embeddings/Regularizer/SquareЦ
(embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_1/embeddings/Regularizer/Const┌
&embedding_1/embeddings/Regularizer/SumSum-embedding_1/embeddings/Regularizer/Square:y:01embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/SumЎ
(embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_1/embeddings/Regularizer/mul/x▄
&embedding_1/embeddings/Regularizer/mulMul1embedding_1/embeddings/Regularizer/mul/x:output:0/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/mulЎ
(embedding_1/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_1/embeddings/Regularizer/add/x┘
&embedding_1/embeddings/Regularizer/addAddV21embedding_1/embeddings/Regularizer/add/x:output:0*embedding_1/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_1/embeddings/Regularizer/add|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         ::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
П
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_22247

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
h
__inference_loss_fn_8_226138
4dense_kernel_regularizer_abs_readvariableop_resource
identityѕ¤
+dense/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp4dense_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:D2*
dtype02-
+dense/kernel/Regularizer/Abs/ReadVariableOpА
dense/kernel/Regularizer/AbsAbs3dense/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:D22
dense/kernel/Regularizer/AbsЉ
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const»
dense/kernel/Regularizer/SumSum dense/kernel/Regularizer/Abs:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/SumЁ
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2 
dense/kernel/Regularizer/mul/x┤
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mulЁ
dense/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dense/kernel/Regularizer/add/x▒
dense/kernel/Regularizer/addAddV2'dense/kernel/Regularizer/add/x:output:0 dense/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/addН
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp4dense_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:D2*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOpГ
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:D22!
dense/kernel/Regularizer/SquareЋ
 dense/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/kernel/Regularizer/Const_1И
dense/kernel/Regularizer/Sum_1Sum#dense/kernel/Regularizer/Square:y:0)dense/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/Sum_1Ѕ
 dense/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<2"
 dense/kernel/Regularizer/mul_1/x╝
dense/kernel/Regularizer/mul_1Mul)dense/kernel/Regularizer/mul_1/x:output:0'dense/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/mul_1░
dense/kernel/Regularizer/add_1AddV2 dense/kernel/Regularizer/add:z:0"dense/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2 
dense/kernel/Regularizer/add_1e
IdentityIdentity"dense/kernel/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Ж
г
D__inference_wide_deep_layer_call_and_return_conditional_losses_22480

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ј:::P L
(
_output_shapes
:         ј
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ч
u
__inference_loss_fn_7_22593E
Aembedding_7_embeddings_regularizer_square_readvariableop_resource
identityѕШ
8embedding_7/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpAembedding_7_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:**
dtype02:
8embedding_7/embeddings/Regularizer/Square/ReadVariableOp╦
)embedding_7/embeddings/Regularizer/SquareSquare@embedding_7/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:*2+
)embedding_7/embeddings/Regularizer/SquareЦ
(embedding_7/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2*
(embedding_7/embeddings/Regularizer/Const┌
&embedding_7/embeddings/Regularizer/SumSum-embedding_7/embeddings/Regularizer/Square:y:01embedding_7/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/SumЎ
(embedding_7/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:2*
(embedding_7/embeddings/Regularizer/mul/x▄
&embedding_7/embeddings/Regularizer/mulMul1embedding_7/embeddings/Regularizer/mul/x:output:0/embedding_7/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/mulЎ
(embedding_7/embeddings/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(embedding_7/embeddings/Regularizer/add/x┘
&embedding_7/embeddings/Regularizer/addAddV21embedding_7/embeddings/Regularizer/add/x:output:0*embedding_7/embeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2(
&embedding_7/embeddings/Regularizer/addm
IdentityIdentity*embedding_7/embeddings/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
Э
~
)__inference_wide_deep_layer_call_fn_22489

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_wide_deep_layer_call_and_return_conditional_losses_207062
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ј::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ј
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┼
`
B__inference_dropout_layer_call_and_return_conditional_losses_20609

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         22

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         22

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs"»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*А
serving_defaultЇ
9
age_in/
serving_default_age_in:0         
K
capital_gain_in8
!serving_default_capital_gain_in:0         
K
capital_loss_in8
!serving_default_capital_loss_in:0         
G
education_inp6
serving_default_education_inp:0         
A

gender_inp3
serving_default_gender_inp:0         
O
hours_per_week_in:
#serving_default_hours_per_week_in:0         
Q
marital_status_inp;
$serving_default_marital_status_inp:0         
Q
native_country_inp;
$serving_default_native_country_inp:0         
I
occupation_inp7
 serving_default_occupation_inp:0         
=
race_inp1
serving_default_race_inp:0         
M
relationship_inp9
"serving_default_relationship_inp:0         
6
wide.
serving_default_wide:0         Щ
G
workclass_inp6
serving_default_workclass_inp:0         =
	wide_deep0
StatefulPartitionedCall:0         tensorflow/serving/predict:└┼
Ў╬
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer_with_weights-4
layer-16
layer_with_weights-5
layer-17
layer_with_weights-6
layer-18
layer_with_weights-7
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer_with_weights-8
layer-26
layer-27
layer_with_weights-9
layer-28
layer-29
layer-30
 layer-31
!layer_with_weights-10
!layer-32
"	optimizer
#	variables
$regularization_losses
%trainable_variables
&	keras_api
'
signatures
Џ__call__
ю_default_save_signature
+Ю&call_and_return_all_conditional_losses"ук
_tf_keras_model╠к{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "workclass_inp"}, "name": "workclass_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "education_inp"}, "name": "education_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "marital_status_inp"}, "name": "marital_status_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "occupation_inp"}, "name": "occupation_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "relationship_inp"}, "name": "relationship_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "race_inp"}, "name": "race_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "gender_inp"}, "name": "gender_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "native_country_inp"}, "name": "native_country_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age_in"}, "name": "age_in", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital_gain_in"}, "name": "capital_gain_in", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital_loss_in"}, "name": "capital_loss_in", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours_per_week_in"}, "name": "hours_per_week_in", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 9, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding", "inbound_nodes": [[["workclass_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 16, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_1", "inbound_nodes": [[["education_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 7, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_2", "inbound_nodes": [[["marital_status_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 15, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_3", "inbound_nodes": [[["occupation_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_4", "inbound_nodes": [[["relationship_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 5, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_5", "inbound_nodes": [[["race_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_6", "inbound_nodes": [[["gender_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 42, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_7", "inbound_nodes": [[["native_country_inp", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape", "inbound_nodes": [[["age_in", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape_1", "inbound_nodes": [[["capital_gain_in", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape_2", "inbound_nodes": [[["capital_loss_in", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape_3", "inbound_nodes": [[["hours_per_week_in", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["embedding", 0, 0, {}], ["embedding_1", 0, 0, {}], ["embedding_2", 0, 0, {}], ["embedding_3", 0, 0, {}], ["embedding_4", 0, 0, {}], ["embedding_5", 0, 0, {}], ["embedding_6", 0, 0, {}], ["embedding_7", 0, 0, {}], ["reshape", 0, 0, {}], ["reshape_1", 0, 0, {}], ["reshape_2", 0, 0, {}], ["reshape_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "deep", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "deep", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 762]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "wide"}, "name": "wide", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["deep", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["wide", 0, 0, {}], ["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "wide_deep", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "wide_deep", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}], "input_layers": [["wide", 0, 0], ["workclass_inp", 0, 0], ["education_inp", 0, 0], ["marital_status_inp", 0, 0], ["occupation_inp", 0, 0], ["relationship_inp", 0, 0], ["race_inp", 0, 0], ["gender_inp", 0, 0], ["native_country_inp", 0, 0], ["age_in", 0, 0], ["capital_gain_in", 0, 0], ["capital_loss_in", 0, 0], ["hours_per_week_in", 0, 0]], "output_layers": [["wide_deep", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 762]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "workclass_inp"}, "name": "workclass_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "education_inp"}, "name": "education_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "marital_status_inp"}, "name": "marital_status_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "occupation_inp"}, "name": "occupation_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "relationship_inp"}, "name": "relationship_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "race_inp"}, "name": "race_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "gender_inp"}, "name": "gender_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "native_country_inp"}, "name": "native_country_inp", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age_in"}, "name": "age_in", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital_gain_in"}, "name": "capital_gain_in", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital_loss_in"}, "name": "capital_loss_in", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours_per_week_in"}, "name": "hours_per_week_in", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 9, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding", "inbound_nodes": [[["workclass_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 16, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_1", "inbound_nodes": [[["education_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 7, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_2", "inbound_nodes": [[["marital_status_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 15, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_3", "inbound_nodes": [[["occupation_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_4", "inbound_nodes": [[["relationship_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 5, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_5", "inbound_nodes": [[["race_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_6", "inbound_nodes": [[["gender_inp", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 42, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "name": "embedding_7", "inbound_nodes": [[["native_country_inp", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape", "inbound_nodes": [[["age_in", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape_1", "inbound_nodes": [[["capital_gain_in", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape_2", "inbound_nodes": [[["capital_loss_in", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}, "name": "reshape_3", "inbound_nodes": [[["hours_per_week_in", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["embedding", 0, 0, {}], ["embedding_1", 0, 0, {}], ["embedding_2", 0, 0, {}], ["embedding_3", 0, 0, {}], ["embedding_4", 0, 0, {}], ["embedding_5", 0, 0, {}], ["embedding_6", 0, 0, {}], ["embedding_7", 0, 0, {}], ["reshape", 0, 0, {}], ["reshape_1", 0, 0, {}], ["reshape_2", 0, 0, {}], ["reshape_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "deep", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "deep", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 762]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "wide"}, "name": "wide", "inbound_nodes": []}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["deep", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["wide", 0, 0, {}], ["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "wide_deep", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "wide_deep", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}], "input_layers": [["wide", 0, 0], ["workclass_inp", 0, 0], ["education_inp", 0, 0], ["marital_status_inp", 0, 0], ["occupation_inp", 0, 0], ["relationship_inp", 0, 0], ["race_inp", 0, 0], ["gender_inp", 0, 0], ["native_country_inp", 0, 0], ["age_in", 0, 0], ["capital_gain_in", 0, 0], ["capital_loss_in", 0, 0], ["hours_per_week_in", 0, 0]], "output_layers": [["wide_deep", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ы"Ь
_tf_keras_input_layer╬{"class_name": "InputLayer", "name": "workclass_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "workclass_inp"}}
ы"Ь
_tf_keras_input_layer╬{"class_name": "InputLayer", "name": "education_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "education_inp"}}
ч"Э
_tf_keras_input_layerп{"class_name": "InputLayer", "name": "marital_status_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "marital_status_inp"}}
з"­
_tf_keras_input_layerл{"class_name": "InputLayer", "name": "occupation_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "occupation_inp"}}
э"З
_tf_keras_input_layerн{"class_name": "InputLayer", "name": "relationship_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "relationship_inp"}}
у"С
_tf_keras_input_layer─{"class_name": "InputLayer", "name": "race_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "race_inp"}}
в"У
_tf_keras_input_layer╚{"class_name": "InputLayer", "name": "gender_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "gender_inp"}}
ч"Э
_tf_keras_input_layerп{"class_name": "InputLayer", "name": "native_country_inp", "dtype": "int32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "int32", "sparse": false, "ragged": false, "name": "native_country_inp"}}
у"С
_tf_keras_input_layer─{"class_name": "InputLayer", "name": "age_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age_in"}}
щ"Ш
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "capital_gain_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital_gain_in"}}
щ"Ш
_tf_keras_input_layerо{"class_name": "InputLayer", "name": "capital_loss_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital_loss_in"}}
§"Щ
_tf_keras_input_layer┌{"class_name": "InputLayer", "name": "hours_per_week_in", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours_per_week_in"}}
┴
(
embeddings
)	variables
*regularization_losses
+trainable_variables
,	keras_api
ъ__call__
+Ъ&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 9, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
к
-
embeddings
.	variables
/regularization_losses
0trainable_variables
1	keras_api
а__call__
+А&call_and_return_all_conditional_losses"Ц
_tf_keras_layerІ{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 16, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
┼
2
embeddings
3	variables
4regularization_losses
5trainable_variables
6	keras_api
б__call__
+Б&call_and_return_all_conditional_losses"ц
_tf_keras_layerі{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 7, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
к
7
embeddings
8	variables
9regularization_losses
:trainable_variables
;	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses"Ц
_tf_keras_layerІ{"class_name": "Embedding", "name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 15, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
┼
<
embeddings
=	variables
>regularization_losses
?trainable_variables
@	keras_api
д__call__
+Д&call_and_return_all_conditional_losses"ц
_tf_keras_layerі{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 6, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
┼
A
embeddings
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
е__call__
+Е&call_and_return_all_conditional_losses"ц
_tf_keras_layerі{"class_name": "Embedding", "name": "embedding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 5, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
┼
F
embeddings
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses"ц
_tf_keras_layerі{"class_name": "Embedding", "name": "embedding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 2, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
к
K
embeddings
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
г__call__
+Г&call_and_return_all_conditional_losses"Ц
_tf_keras_layerІ{"class_name": "Embedding", "name": "embedding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "config": {"name": "embedding_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 42, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
¤
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
«__call__
+»&call_and_return_all_conditional_losses"Й
_tf_keras_layerц{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}}
М
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"┬
_tf_keras_layerе{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}}
М
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"┬
_tf_keras_layerе{"class_name": "Reshape", "name": "reshape_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_2", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}}
М
\	variables
]regularization_losses
^trainable_variables
_	keras_api
┤__call__
+х&call_and_return_all_conditional_losses"┬
_tf_keras_layerе{"class_name": "Reshape", "name": "reshape_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [1, 1]}}}
╚
`	variables
aregularization_losses
btrainable_variables
c	keras_api
Х__call__
+и&call_and_return_all_conditional_losses"и
_tf_keras_layerЮ{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 8]}, {"class_name": "TensorShape", "items": [null, 1, 1]}, {"class_name": "TensorShape", "items": [null, 1, 1]}, {"class_name": "TensorShape", "items": [null, 1, 1]}, {"class_name": "TensorShape", "items": [null, 1, 1]}]}
┴
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
И__call__
+╣&call_and_return_all_conditional_losses"░
_tf_keras_layerќ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
А

hkernel
ibias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
║__call__
+╗&call_and_return_all_conditional_losses"Щ
_tf_keras_layerЯ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 68}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 68]}}
└
n	variables
oregularization_losses
ptrainable_variables
q	keras_api
╝__call__
+й&call_and_return_all_conditional_losses"»
_tf_keras_layerЋ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
╔

rkernel
sbias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
Й__call__
+┐&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"class_name": "Dense", "name": "deep", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "deep", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
у"С
_tf_keras_input_layer─{"class_name": "InputLayer", "name": "wide", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 762]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 762]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "wide"}}
─
x	variables
yregularization_losses
ztrainable_variables
{	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"│
_tf_keras_layerЎ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
Г
|	variables
}regularization_losses
~trainable_variables
	keras_api
┬__call__
+├&call_and_return_all_conditional_losses"ю
_tf_keras_layerѓ{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 762]}, {"class_name": "TensorShape", "items": [null, 20]}]}
П
ђkernel
	Ђbias
ѓ	variables
Ѓregularization_losses
ёtrainable_variables
Ё	keras_api
─__call__
+┼&call_and_return_all_conditional_losses"░
_tf_keras_layerќ{"class_name": "Dense", "name": "wide_deep", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "wide_deep", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 782}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 782]}}
З
	єiter
Єbeta_1
ѕbeta_2

Ѕdecay
іlearning_rate(m -mђ2mЂ7mѓ<mЃAmёFmЁKmєhmЄimѕrmЅsmі	ђmІ	Ђmї(vЇ-vј2vЈ7vљ<vЉAvњFvЊKvћhvЋivќrvЌsvў	ђvЎ	Ђvџ"
	optimizer
ѕ
(0
-1
22
73
<4
A5
F6
K7
h8
i9
r10
s11
ђ12
Ђ13"
trackable_list_wrapper
h
к0
К1
╚2
╔3
╩4
╦5
╠6
═7
╬8"
trackable_list_wrapper
ѕ
(0
-1
22
73
<4
A5
F6
K7
h8
i9
r10
s11
ђ12
Ђ13"
trackable_list_wrapper
М
Іmetrics
 їlayer_regularization_losses
#	variables
Їlayers
јnon_trainable_variables
Јlayer_metrics
$regularization_losses
%trainable_variables
Џ__call__
ю_default_save_signature
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
-
¤serving_default"
signature_map
&:$	2embedding/embeddings
'
(0"
trackable_list_wrapper
(
к0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
х
љmetrics
 Љlayer_regularization_losses
)	variables
њlayers
Њnon_trainable_variables
ћlayer_metrics
*regularization_losses
+trainable_variables
ъ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
(:&2embedding_1/embeddings
'
-0"
trackable_list_wrapper
(
К0"
trackable_list_wrapper
'
-0"
trackable_list_wrapper
х
Ћmetrics
 ќlayer_regularization_losses
.	variables
Ќlayers
ўnon_trainable_variables
Ўlayer_metrics
/regularization_losses
0trainable_variables
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
(:&2embedding_2/embeddings
'
20"
trackable_list_wrapper
(
╚0"
trackable_list_wrapper
'
20"
trackable_list_wrapper
х
џmetrics
 Џlayer_regularization_losses
3	variables
юlayers
Юnon_trainable_variables
ъlayer_metrics
4regularization_losses
5trainable_variables
б__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
(:&2embedding_3/embeddings
'
70"
trackable_list_wrapper
(
╔0"
trackable_list_wrapper
'
70"
trackable_list_wrapper
х
Ъmetrics
 аlayer_regularization_losses
8	variables
Аlayers
бnon_trainable_variables
Бlayer_metrics
9regularization_losses
:trainable_variables
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
(:&2embedding_4/embeddings
'
<0"
trackable_list_wrapper
(
╩0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
х
цmetrics
 Цlayer_regularization_losses
=	variables
дlayers
Дnon_trainable_variables
еlayer_metrics
>regularization_losses
?trainable_variables
д__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
(:&2embedding_5/embeddings
'
A0"
trackable_list_wrapper
(
╦0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
х
Еmetrics
 фlayer_regularization_losses
B	variables
Фlayers
гnon_trainable_variables
Гlayer_metrics
Cregularization_losses
Dtrainable_variables
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
(:&2embedding_6/embeddings
'
F0"
trackable_list_wrapper
(
╠0"
trackable_list_wrapper
'
F0"
trackable_list_wrapper
х
«metrics
 »layer_regularization_losses
G	variables
░layers
▒non_trainable_variables
▓layer_metrics
Hregularization_losses
Itrainable_variables
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
(:&*2embedding_7/embeddings
'
K0"
trackable_list_wrapper
(
═0"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
х
│metrics
 ┤layer_regularization_losses
L	variables
хlayers
Хnon_trainable_variables
иlayer_metrics
Mregularization_losses
Ntrainable_variables
г__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Иmetrics
 ╣layer_regularization_losses
P	variables
║layers
╗non_trainable_variables
╝layer_metrics
Qregularization_losses
Rtrainable_variables
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
йmetrics
 Йlayer_regularization_losses
T	variables
┐layers
└non_trainable_variables
┴layer_metrics
Uregularization_losses
Vtrainable_variables
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
┬metrics
 ├layer_regularization_losses
X	variables
─layers
┼non_trainable_variables
кlayer_metrics
Yregularization_losses
Ztrainable_variables
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Кmetrics
 ╚layer_regularization_losses
\	variables
╔layers
╩non_trainable_variables
╦layer_metrics
]regularization_losses
^trainable_variables
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
╠metrics
 ═layer_regularization_losses
`	variables
╬layers
¤non_trainable_variables
лlayer_metrics
aregularization_losses
btrainable_variables
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Лmetrics
 мlayer_regularization_losses
d	variables
Мlayers
нnon_trainable_variables
Нlayer_metrics
eregularization_losses
ftrainable_variables
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
:D22dense/kernel
:22
dense/bias
.
h0
i1"
trackable_list_wrapper
(
╬0"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
х
оmetrics
 Оlayer_regularization_losses
j	variables
пlayers
┘non_trainable_variables
┌layer_metrics
kregularization_losses
ltrainable_variables
║__call__
+╗&call_and_return_all_conditional_losses
'╗"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
█metrics
 ▄layer_regularization_losses
n	variables
Пlayers
яnon_trainable_variables
▀layer_metrics
oregularization_losses
ptrainable_variables
╝__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
:22deep/kernel
:2	deep/bias
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
х
Яmetrics
 рlayer_regularization_losses
t	variables
Рlayers
сnon_trainable_variables
Сlayer_metrics
uregularization_losses
vtrainable_variables
Й__call__
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
тmetrics
 Тlayer_regularization_losses
x	variables
уlayers
Уnon_trainable_variables
жlayer_metrics
yregularization_losses
ztrainable_variables
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Жmetrics
 вlayer_regularization_losses
|	variables
Вlayers
ьnon_trainable_variables
Ьlayer_metrics
}regularization_losses
~trainable_variables
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
#:!	ј2wide_deep/kernel
:2wide_deep/bias
0
ђ0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ђ0
Ђ1"
trackable_list_wrapper
И
№metrics
 ­layer_regularization_losses
ѓ	variables
ыlayers
Ыnon_trainable_variables
зlayer_metrics
Ѓregularization_losses
ёtrainable_variables
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
З0
ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
ъ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
к0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
К0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
╚0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
╔0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
╩0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
╦0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
╠0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
═0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
╬0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┐

Шtotal

эcount
Э	variables
щ	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 

Щtotal

чcount
Ч
_fn_kwargs
§	variables
■	keras_api"│
_tf_keras_metricў{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
Ш0
э1"
trackable_list_wrapper
.
Э	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Щ0
ч1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
+:)	2Adam/embedding/embeddings/m
-:+2Adam/embedding_1/embeddings/m
-:+2Adam/embedding_2/embeddings/m
-:+2Adam/embedding_3/embeddings/m
-:+2Adam/embedding_4/embeddings/m
-:+2Adam/embedding_5/embeddings/m
-:+2Adam/embedding_6/embeddings/m
-:+*2Adam/embedding_7/embeddings/m
#:!D22Adam/dense/kernel/m
:22Adam/dense/bias/m
": 22Adam/deep/kernel/m
:2Adam/deep/bias/m
(:&	ј2Adam/wide_deep/kernel/m
!:2Adam/wide_deep/bias/m
+:)	2Adam/embedding/embeddings/v
-:+2Adam/embedding_1/embeddings/v
-:+2Adam/embedding_2/embeddings/v
-:+2Adam/embedding_3/embeddings/v
-:+2Adam/embedding_4/embeddings/v
-:+2Adam/embedding_5/embeddings/v
-:+2Adam/embedding_6/embeddings/v
-:+*2Adam/embedding_7/embeddings/v
#:!D22Adam/dense/kernel/v
:22Adam/dense/bias/v
": 22Adam/deep/kernel/v
:2Adam/deep/bias/v
(:&	ј2Adam/wide_deep/kernel/v
!:2Adam/wide_deep/bias/v
Р2▀
%__inference_model_layer_call_fn_21322
%__inference_model_layer_call_fn_21915
%__inference_model_layer_call_fn_21134
%__inference_model_layer_call_fn_21960└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
п2Н
 __inference__wrapped_model_20168░
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *ЪбЏ
ўџћ
і
wide         Щ
'і$
workclass_inp         
'і$
education_inp         
,і)
marital_status_inp         
(і%
occupation_inp         
*і'
relationship_inp         
"і
race_inp         
$і!

gender_inp         
,і)
native_country_inp         
 і
age_in         
)і&
capital_gain_in         
)і&
capital_loss_in         
+і(
hours_per_week_in         
╬2╦
@__inference_model_layer_call_and_return_conditional_losses_20945
@__inference_model_layer_call_and_return_conditional_losses_20802
@__inference_model_layer_call_and_return_conditional_losses_21670
@__inference_model_layer_call_and_return_conditional_losses_21870└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_embedding_layer_call_fn_21992б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_embedding_layer_call_and_return_conditional_losses_21985б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_embedding_1_layer_call_fn_22024б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_embedding_1_layer_call_and_return_conditional_losses_22017б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_embedding_2_layer_call_fn_22056б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_embedding_2_layer_call_and_return_conditional_losses_22049б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_embedding_3_layer_call_fn_22088б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_embedding_3_layer_call_and_return_conditional_losses_22081б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_embedding_4_layer_call_fn_22120б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_embedding_4_layer_call_and_return_conditional_losses_22113б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_embedding_5_layer_call_fn_22152б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_embedding_5_layer_call_and_return_conditional_losses_22145б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_embedding_6_layer_call_fn_22184б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_embedding_6_layer_call_and_return_conditional_losses_22177б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_embedding_7_layer_call_fn_22216б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_embedding_7_layer_call_and_return_conditional_losses_22209б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_reshape_layer_call_fn_22234б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_reshape_layer_call_and_return_conditional_losses_22229б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_reshape_1_layer_call_fn_22252б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_reshape_1_layer_call_and_return_conditional_losses_22247б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_reshape_2_layer_call_fn_22270б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_reshape_2_layer_call_and_return_conditional_losses_22265б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_reshape_3_layer_call_fn_22288б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_reshape_3_layer_call_and_return_conditional_losses_22283б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_concatenate_layer_call_fn_22321б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_concatenate_layer_call_and_return_conditional_losses_22305б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_flatten_layer_call_fn_22332б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_flatten_layer_call_and_return_conditional_losses_22327б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_dense_layer_call_fn_22382б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_dense_layer_call_and_return_conditional_losses_22373б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ї2Ѕ
'__inference_dropout_layer_call_fn_22409
'__inference_dropout_layer_call_fn_22404┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┬2┐
B__inference_dropout_layer_call_and_return_conditional_losses_22394
B__inference_dropout_layer_call_and_return_conditional_losses_22399┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╬2╦
$__inference_deep_layer_call_fn_22429б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ж2Т
?__inference_deep_layer_call_and_return_conditional_losses_22420б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љ2Ї
)__inference_dropout_1_layer_call_fn_22456
)__inference_dropout_1_layer_call_fn_22451┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
к2├
D__inference_dropout_1_layer_call_and_return_conditional_losses_22446
D__inference_dropout_1_layer_call_and_return_conditional_losses_22441┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
О2н
-__inference_concatenate_1_layer_call_fn_22469б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_concatenate_1_layer_call_and_return_conditional_losses_22463б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_wide_deep_layer_call_fn_22489б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_wide_deep_layer_call_and_return_conditional_losses_22480б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▓2»
__inference_loss_fn_0_22502Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_1_22515Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_2_22528Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_3_22541Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_4_22554Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_5_22567Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_6_22580Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_7_22593Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
▓2»
__inference_loss_fn_8_22613Ј
Є▓Ѓ
FullArgSpec
argsџ 
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *б 
жBТ
#__inference_signature_wrapper_21456age_incapital_gain_incapital_loss_ineducation_inp
gender_inphours_per_week_inmarital_status_inpnative_country_inpoccupation_inprace_inprelationship_inpwideworkclass_inpю
 __inference__wrapped_model_20168э(-27<AFKhirsђЂФбД
ЪбЏ
ўџћ
і
wide         Щ
'і$
workclass_inp         
'і$
education_inp         
,і)
marital_status_inp         
(і%
occupation_inp         
*і'
relationship_inp         
"і
race_inp         
$і!

gender_inp         
,і)
native_country_inp         
 і
age_in         
)і&
capital_gain_in         
)і&
capital_loss_in         
+і(
hours_per_week_in         
ф "5ф2
0
	wide_deep#і 
	wide_deep         м
H__inference_concatenate_1_layer_call_and_return_conditional_losses_22463Ё[бX
QбN
LџI
#і 
inputs/0         Щ
"і
inputs/1         
ф "&б#
і
0         ј
џ Е
-__inference_concatenate_1_layer_call_fn_22469x[бX
QбN
LџI
#і 
inputs/0         Щ
"і
inputs/1         
ф "і         јЫ
F__inference_concatenate_layer_call_and_return_conditional_losses_22305Дщбш
ьбж
ТџР
&і#
inputs/0         
&і#
inputs/1         
&і#
inputs/2         
&і#
inputs/3         
&і#
inputs/4         
&і#
inputs/5         
&і#
inputs/6         
&і#
inputs/7         
&і#
inputs/8         
&і#
inputs/9         
'і$
	inputs/10         
'і$
	inputs/11         
ф ")б&
і
0         D
џ ╩
+__inference_concatenate_layer_call_fn_22321џщбш
ьбж
ТџР
&і#
inputs/0         
&і#
inputs/1         
&і#
inputs/2         
&і#
inputs/3         
&і#
inputs/4         
&і#
inputs/5         
&і#
inputs/6         
&і#
inputs/7         
&і#
inputs/8         
&і#
inputs/9         
'і$
	inputs/10         
'і$
	inputs/11         
ф "і         DЪ
?__inference_deep_layer_call_and_return_conditional_losses_22420\rs/б,
%б"
 і
inputs         2
ф "%б"
і
0         
џ w
$__inference_deep_layer_call_fn_22429Ors/б,
%б"
 і
inputs         2
ф "і         а
@__inference_dense_layer_call_and_return_conditional_losses_22373\hi/б,
%б"
 і
inputs         D
ф "%б"
і
0         2
џ x
%__inference_dense_layer_call_fn_22382Ohi/б,
%б"
 і
inputs         D
ф "і         2ц
D__inference_dropout_1_layer_call_and_return_conditional_losses_22441\3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ ц
D__inference_dropout_1_layer_call_and_return_conditional_losses_22446\3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ |
)__inference_dropout_1_layer_call_fn_22451O3б0
)б&
 і
inputs         
p
ф "і         |
)__inference_dropout_1_layer_call_fn_22456O3б0
)б&
 і
inputs         
p 
ф "і         б
B__inference_dropout_layer_call_and_return_conditional_losses_22394\3б0
)б&
 і
inputs         2
p
ф "%б"
і
0         2
џ б
B__inference_dropout_layer_call_and_return_conditional_losses_22399\3б0
)б&
 і
inputs         2
p 
ф "%б"
і
0         2
џ z
'__inference_dropout_layer_call_fn_22404O3б0
)б&
 і
inputs         2
p
ф "і         2z
'__inference_dropout_layer_call_fn_22409O3б0
)б&
 і
inputs         2
p 
ф "і         2Е
F__inference_embedding_1_layer_call_and_return_conditional_losses_22017_-/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ Ђ
+__inference_embedding_1_layer_call_fn_22024R-/б,
%б"
 і
inputs         
ф "і         Е
F__inference_embedding_2_layer_call_and_return_conditional_losses_22049_2/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ Ђ
+__inference_embedding_2_layer_call_fn_22056R2/б,
%б"
 і
inputs         
ф "і         Е
F__inference_embedding_3_layer_call_and_return_conditional_losses_22081_7/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ Ђ
+__inference_embedding_3_layer_call_fn_22088R7/б,
%б"
 і
inputs         
ф "і         Е
F__inference_embedding_4_layer_call_and_return_conditional_losses_22113_</б,
%б"
 і
inputs         
ф ")б&
і
0         
џ Ђ
+__inference_embedding_4_layer_call_fn_22120R</б,
%б"
 і
inputs         
ф "і         Е
F__inference_embedding_5_layer_call_and_return_conditional_losses_22145_A/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ Ђ
+__inference_embedding_5_layer_call_fn_22152RA/б,
%б"
 і
inputs         
ф "і         Е
F__inference_embedding_6_layer_call_and_return_conditional_losses_22177_F/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ Ђ
+__inference_embedding_6_layer_call_fn_22184RF/б,
%б"
 і
inputs         
ф "і         Е
F__inference_embedding_7_layer_call_and_return_conditional_losses_22209_K/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ Ђ
+__inference_embedding_7_layer_call_fn_22216RK/б,
%б"
 і
inputs         
ф "і         Д
D__inference_embedding_layer_call_and_return_conditional_losses_21985_(/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ 
)__inference_embedding_layer_call_fn_21992R(/б,
%б"
 і
inputs         
ф "і         б
B__inference_flatten_layer_call_and_return_conditional_losses_22327\3б0
)б&
$і!
inputs         D
ф "%б"
і
0         D
џ z
'__inference_flatten_layer_call_fn_22332O3б0
)б&
$і!
inputs         D
ф "і         D:
__inference_loss_fn_0_22502(б

б 
ф "і :
__inference_loss_fn_1_22515-б

б 
ф "і :
__inference_loss_fn_2_225282б

б 
ф "і :
__inference_loss_fn_3_225417б

б 
ф "і :
__inference_loss_fn_4_22554<б

б 
ф "і :
__inference_loss_fn_5_22567Aб

б 
ф "і :
__inference_loss_fn_6_22580Fб

б 
ф "і :
__inference_loss_fn_7_22593Kб

б 
ф "і :
__inference_loss_fn_8_22613hб

б 
ф "і ┤
@__inference_model_layer_call_and_return_conditional_losses_20802№(-27<AFKhirsђЂ│б»
ДбБ
ўџћ
і
wide         Щ
'і$
workclass_inp         
'і$
education_inp         
,і)
marital_status_inp         
(і%
occupation_inp         
*і'
relationship_inp         
"і
race_inp         
$і!

gender_inp         
,і)
native_country_inp         
 і
age_in         
)і&
capital_gain_in         
)і&
capital_loss_in         
+і(
hours_per_week_in         
p

 
ф "%б"
і
0         
џ ┤
@__inference_model_layer_call_and_return_conditional_losses_20945№(-27<AFKhirsђЂ│б»
ДбБ
ўџћ
і
wide         Щ
'і$
workclass_inp         
'і$
education_inp         
,і)
marital_status_inp         
(і%
occupation_inp         
*і'
relationship_inp         
"і
race_inp         
$і!

gender_inp         
,і)
native_country_inp         
 і
age_in         
)і&
capital_gain_in         
)і&
capital_loss_in         
+і(
hours_per_week_in         
p 

 
ф "%б"
і
0         
џ Э
@__inference_model_layer_call_and_return_conditional_losses_21670│(-27<AFKhirsђЂэбз
вбу
▄џп
#і 
inputs/0         Щ
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
#і 
	inputs/11         
#і 
	inputs/12         
p

 
ф "%б"
і
0         
џ Э
@__inference_model_layer_call_and_return_conditional_losses_21870│(-27<AFKhirsђЂэбз
вбу
▄џп
#і 
inputs/0         Щ
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
#і 
	inputs/11         
#і 
	inputs/12         
p 

 
ф "%б"
і
0         
џ ї
%__inference_model_layer_call_fn_21134Р(-27<AFKhirsђЂ│б»
ДбБ
ўџћ
і
wide         Щ
'і$
workclass_inp         
'і$
education_inp         
,і)
marital_status_inp         
(і%
occupation_inp         
*і'
relationship_inp         
"і
race_inp         
$і!

gender_inp         
,і)
native_country_inp         
 і
age_in         
)і&
capital_gain_in         
)і&
capital_loss_in         
+і(
hours_per_week_in         
p

 
ф "і         ї
%__inference_model_layer_call_fn_21322Р(-27<AFKhirsђЂ│б»
ДбБ
ўџћ
і
wide         Щ
'і$
workclass_inp         
'і$
education_inp         
,і)
marital_status_inp         
(і%
occupation_inp         
*і'
relationship_inp         
"і
race_inp         
$і!

gender_inp         
,і)
native_country_inp         
 і
age_in         
)і&
capital_gain_in         
)і&
capital_loss_in         
+і(
hours_per_week_in         
p 

 
ф "і         л
%__inference_model_layer_call_fn_21915д(-27<AFKhirsђЂэбз
вбу
▄џп
#і 
inputs/0         Щ
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
#і 
	inputs/11         
#і 
	inputs/12         
p

 
ф "і         л
%__inference_model_layer_call_fn_21960д(-27<AFKhirsђЂэбз
вбу
▄џп
#і 
inputs/0         Щ
"і
inputs/1         
"і
inputs/2         
"і
inputs/3         
"і
inputs/4         
"і
inputs/5         
"і
inputs/6         
"і
inputs/7         
"і
inputs/8         
"і
inputs/9         
#і 
	inputs/10         
#і 
	inputs/11         
#і 
	inputs/12         
p 

 
ф "і         ц
D__inference_reshape_1_layer_call_and_return_conditional_losses_22247\/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ |
)__inference_reshape_1_layer_call_fn_22252O/б,
%б"
 і
inputs         
ф "і         ц
D__inference_reshape_2_layer_call_and_return_conditional_losses_22265\/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ |
)__inference_reshape_2_layer_call_fn_22270O/б,
%б"
 і
inputs         
ф "і         ц
D__inference_reshape_3_layer_call_and_return_conditional_losses_22283\/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ |
)__inference_reshape_3_layer_call_fn_22288O/б,
%б"
 і
inputs         
ф "і         б
B__inference_reshape_layer_call_and_return_conditional_losses_22229\/б,
%б"
 і
inputs         
ф ")б&
і
0         
џ z
'__inference_reshape_layer_call_fn_22234O/б,
%б"
 і
inputs         
ф "і         з
#__inference_signature_wrapper_21456╦(-27<AFKhirsђЂ бч
б 
зф№
*
age_in і
age_in         
<
capital_gain_in)і&
capital_gain_in         
<
capital_loss_in)і&
capital_loss_in         
8
education_inp'і$
education_inp         
2

gender_inp$і!

gender_inp         
@
hours_per_week_in+і(
hours_per_week_in         
B
marital_status_inp,і)
marital_status_inp         
B
native_country_inp,і)
native_country_inp         
:
occupation_inp(і%
occupation_inp         
.
race_inp"і
race_inp         
>
relationship_inp*і'
relationship_inp         
'
wideі
wide         Щ
8
workclass_inp'і$
workclass_inp         "5ф2
0
	wide_deep#і 
	wide_deep         Д
D__inference_wide_deep_layer_call_and_return_conditional_losses_22480_ђЂ0б-
&б#
!і
inputs         ј
ф "%б"
і
0         
џ 
)__inference_wide_deep_layer_call_fn_22489RђЂ0б-
&б#
!і
inputs         ј
ф "і         