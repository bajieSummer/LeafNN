[[Debug]:[TestSpecialNT]isPasStr=caseSucceed case_f=y = 2*log10(3.0x+4)-20:
 initX=[5]
,lastfx=-inf,expectLessF=1e-15,iterNum=3
,X=
[[-1.3333333333333333]]
,grad=
[[0.]]
]
[[Debug]:[TestSpecialNT]isPasStr=caseSucceed case_f=y = 20*ln(1000x+4)-30:
 initX=[100]
,lastfx=-inf,expectLessF=1e-15,iterNum=3
,X=
[[-0.004]]
,grad=
[[0.]]
]
[[Debug]:[TestSpecialNT]isPasStr=caseSucceed case_f=y = -20*ln(1000x+4)-30:
 initX=[100]
,lastfx=-1149.303273479762,expectLessF=-1000,iterNum=200
,X=
[[2.0200483406285727e+21]]
,grad=
[[-9.900753164043914e-21]]
]
[[Debug]:[TestSpecialNT]isPasStr=caseSucceed case_f=y = -20*ln(1000x+y+4)-30:
 initX=[1, 100]
,lastfx=-625.6921984784741,expectLessF=-500,iterNum=200
,X=
[[8.637686727200470e+09]
 [1.187450215835216e+09]]
,grad=
[[-2.321337354631299e-09]
 [-2.321337354631299e-12]]
]
[[Debug]:[TestSpecialNT]isPasStr=caseSucceed case_f=y = -20*ln(1000x1+x2+100x3+4)-30:
 initX=[0.0, 100.0, 10.0]
,lastfx=-625.9248407286855,expectLessF=-500,iterNum=200
,X=
[[8.5504191258258476e+09]
 [1.9025323236699373e+07]
 [1.8920456217767565e+09]]
,grad=
[[-2.2944917356744913e-09]
 [-2.2944917356744912e-12]
 [-2.2944917356744913e-10]]
]
[[Debug]:[TestSpecialNT]runLogLinearTestCases-->End]