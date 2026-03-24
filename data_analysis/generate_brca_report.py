import pandas as pd
import numpy as np
import os
import base64
import weasyprint

D = './output_brca_patients/'

def img_b64(path):
    if not os.path.exists(path): return ''
    with open(path,'rb') as f: return base64.b64encode(f.read()).decode()

def img(path, w='100%'):
    b = img_b64(path)
    return f'<img src="data:image/png;base64,{b}" style="width:{w};">' if b else '<p><em>(이미지 없음)</em></p>'

def tbl(df):
    h = '<table><thead><tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr></thead><tbody>'
    for _,row in df.iterrows():
        h += '<tr>'
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                v = f'{v:.2e}' if (abs(v)<0.001 and v!=0) else (f'{v:,.1f}' if abs(v)>100 else f'{v:.4f}')
            h += f'<td>{v}</td>'
        h += '</tr>'
    h += '</tbody></table>'
    return h

# Load data
overview = pd.read_csv(f'{D}01_BRCA_Data_Overview.csv')
desc = pd.read_csv(f'{D}02_BRCA_Descriptive_Stats.csv').rename(columns={'Unnamed: 0':'Gene'})
ttest = pd.read_csv(f'{D}02_BRCA_TTest_Results.csv')
pb = pd.read_csv(f'{D}03_BRCA_Point_Biserial.csv')
chi = pd.read_csv(f'{D}05_BRCA_ChiSquare.csv')
tss = pd.read_csv(f'{D}05_BRCA_TSS_Distribution.csv')
top20 = pd.read_csv(f'{D}08_BRCA_Top20_Genes.csv')
brca_det = pd.read_csv(f'{D}09_BRCA1_BRCA2_Detail.csv')
quartile = pd.read_csv(f'{D}09_BRCA1_Quartile_Analysis.csv')
vs_other = pd.read_csv(f'{D}10_BRCA_vs_OtherCancers.csv')

html = """<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<style>
@page { size: A4; margin: 2cm 2.5cm; }
body { font-family: 'Noto Sans KR','Malgun Gothic',sans-serif; font-size: 10.5pt; line-height: 1.7; color: #222; }
h1 { font-size: 22pt; font-weight: 900; color: #111; margin-bottom: 0.3em; }
h2 { font-size: 16pt; font-weight: 800; margin-top: 1.8em; color: #111; page-break-after: avoid; }
h3 { font-size: 13pt; font-weight: 700; margin-top: 1.3em; color: #222; page-break-after: avoid; }
table { width: 100%; border-collapse: collapse; margin: 0.8em 0 1.2em; font-size: 9.5pt; }
th { background: #f5f5f5; border: 1px solid #ddd; padding: 6px 10px; text-align: center; font-weight: 700; }
td { border: 1px solid #ddd; padding: 5px 10px; text-align: center; }
tr:nth-child(even) { background: #fafafa; }
.info-box { background: #f8f9fa; border-left: 4px solid #4a90d9; padding: 12px 16px; margin: 1em 0; font-size: 10pt; }
.insight { background: #fff8e6; border-left: 4px solid #e6a817; padding: 10px 14px; margin: 0.8em 0; font-size: 9.5pt; }
.toc ol { line-height: 2.2; }
.toc a { color: #0077cc; text-decoration: none; }
img { max-width: 100%; display: block; margin: 0.5em auto; }
.page-break { page-break-before: always; }
</style></head><body>

<h1>TCGA-BRCA 유방암 환자 통계 분석 보고서</h1>
<div class="info-box">
<strong>분석 일자</strong>: 2026-03-18<br>
<strong>데이터셋</strong>: GSE62944 (TCGA RNA-Seq TPM) 중 BRCA 프로젝트 샘플 추출<br>
<strong>분석 대상</strong>: 종양 1,105명 + 정상 113명 = <strong>총 1,218명</strong> / 23,368개 유전자 전수조사<br>
<strong>타겟 비율</strong>: Tumor : Normal = 90.7% : 9.3%
</div>

<div class="toc"><h2>목차</h2><ol>
<li><a href="#s1">데이터 개요</a></li>
<li><a href="#s2">기술통계</a></li>
<li><a href="#s3">상관분석</a></li>
<li><a href="#s4">교호작용 분석</a></li>
<li><a href="#s5">교차분석 &amp; 카이제곱 검정</a></li>
<li><a href="#s6">분포 시각화</a></li>
<li><a href="#s7">전수조사 결과 (Top 20 유전자)</a></li>
<li><a href="#s8">BRCA1/BRCA2 심층 분석</a></li>
<li><a href="#s9">BRCA vs 다른 암종 비교</a></li>
<li><a href="#s10">종합 요약 및 시사점</a></li>
</ol></div>
"""

# === 1 ===
html += '<h2 id="s1" class="page-break">1. 데이터 개요</h2>'
html += """<p>GSE62944 전체 10,005개 샘플 중 TCGA 바코드의 TSS(Tissue Source Site) 코드를 이용하여
TCGA-BRCA(Breast Invasive Carcinoma) 프로젝트에 해당하는 <strong>1,218개 샘플</strong>을 추출했습니다.</p>"""

html += '<h3>BRCA 환자 데이터 요약</h3>'
html += """<table><thead><tr><th>항목</th><th>값</th><th>설명</th></tr></thead><tbody>
<tr><td>BRCA 종양 샘플</td><td><strong>1,105명</strong></td><td>유방 침윤성 암종</td></tr>
<tr><td>BRCA 정상 샘플</td><td><strong>113명</strong></td><td>유방 정상 조직</td></tr>
<tr><td>전체 BRCA 샘플</td><td><strong>1,218명</strong></td><td>종양 + 정상</td></tr>
<tr><td>유전자 수</td><td>23,368</td><td>전수조사 완료</td></tr>
<tr><td>TSS 기관 수</td><td>31개</td><td>유방암 조직 수집 기관</td></tr>
<tr><td>타겟 비율</td><td>90.7% : 9.3%</td><td>종양 : 정상</td></tr>
</tbody></table>"""

html += """<div class="insight"><strong>핵심 인사이트</strong>: 전체 TCGA 데이터(10,005 샘플) 중 유방암은
<strong>12.2%</strong>(1,218명)로 단일 암종 중 두 번째로 큰 코호트입니다.
정상 조직이 113명으로 충분한 대조군을 확보하고 있습니다.</div>"""

# === 2 ===
html += '<h2 id="s2" class="page-break">2. 기술통계</h2>'
html += '<h3>2-1. 주요 유전자 기술통계 요약</h3>'

desc_show = desc[['Gene','count','mean','std','25%','50%','75%','max','Skewness','Kurtosis']]
html += tbl(desc_show)

html += """<div class="insight"><strong>핵심 인사이트</strong>:
<ul style="margin:0.3em 0;">
<li><strong>GATA3</strong>: 유방암에서 평균 514 TPM으로 가장 높은 발현량 — 유방암 마커로서의 역할 확인</li>
<li><strong>CDH1</strong>(E-cadherin): 평균 302 TPM — 침윤성 소엽암(ILC)에서 손실되는 핵심 유전자</li>
<li><strong>ESR1</strong>(에스트로겐 수용체): 평균 207 TPM, 왜도 1.74 — ER+ / ER- 서브타입 구분 가능</li>
<li><strong>PIK3CA</strong>: 왜도 24.3, 첨도 747 — <strong>극심한 이상치</strong> 존재, 돌연변이 샘플 의심</li>
</ul></div>"""

html += '<h3>2-2. Tumor vs Normal T-검정 결과</h3>'
html += tbl(ttest)

html += """<div class="insight"><strong>핵심 인사이트</strong>:
<ul style="margin:0.3em 0;">
<li><strong>BRCA2</strong>: 종양에서 정상 대비 <strong>2.6배</strong> 과발현 (t=14.64, p≈0)</li>
<li><strong>BRCA1</strong>: 종양에서 <strong>2.0배</strong> 과발현 (t=12.68, p≈0) — DNA 손상 복구 활성화 반영</li>
<li><strong>GATA3</strong>: 종양에서 <strong>3.2배</strong> 과발현 — 유방암 특이적 전사인자</li>
<li><strong>EGFR</strong>: 종양에서 정상 대비 <strong>0.41배로 감소</strong> (t=-29.7) — 유방암에서 하향조절</li>
<li><strong>MYC</strong>: 종양에서 <strong>0.47배로 감소</strong> — 유방암 정상 조직이 오히려 MYC 발현이 높음</li>
<li><strong>TP53</strong>: p=0.49로 <strong>유의하지 않음</strong> — 유방암에서 TP53 발현량 차이는 미미</li>
</ul></div>"""

# === 3 ===
html += '<h2 id="s3" class="page-break">3. 상관분석</h2>'
html += '<h3>3-1. Pearson & Spearman 상관계수 히트맵</h3>'
html += img(f'{D}03_BRCA_Correlation_Heatmaps.png')

html += '<h3>3-2. Point-Biserial 상관 (종양 여부 vs 유전자 발현)</h3>'
pb_show = pb.copy()
pb_show.columns = ['유전자', 'PB 상관계수', 'p-value']
html += tbl(pb_show)

html += """<div class="insight"><strong>핵심 인사이트</strong>: 유방암 환자 내에서 종양 예측에 가장 강한 유전자는:
<ol style="margin:0.3em 0;">
<li><strong>EGFR</strong> (r=-0.477): 종양에서 <strong>강하게 억제</strong></li>
<li><strong>MYC</strong> (r=-0.307): 종양에서 하향조절</li>
<li><strong>PTEN</strong> (r=-0.305): 종양 억제 유전자 역할 확인</li>
<li><strong>BRCA2</strong> (r=+0.269): 종양에서 과발현</li>
<li><strong>PIK3CA</strong> (r=-0.263): 종양에서 하향조절</li>
</ol>
TP53 (r=-0.016)는 <strong>유방암 내에서는 종양/정상 구분력이 없음</strong>을 확인했습니다.</div>"""

# === 4 ===
html += '<h2 id="s4" class="page-break">4. 교호작용 분석</h2>'

html += '<h3>4-1. TSS Code x 종양 여부 → BRCA1 발현</h3>'
html += img(f'{D}04_Interaction_TSS_BRCA1.png')

html += '<h3>4-2. TSS Code x 종양 여부 → BRCA2 발현</h3>'
html += img(f'{D}04_Interaction_TSS_BRCA2.png')

html += '<h3>4-3. TSS Code x 종양 여부 → ESR1 (에스트로겐 수용체) 발현</h3>'
html += img(f'{D}04_Interaction_TSS_ESR1.png')

html += '<h3>4-4. TSS Code x BRCA1 발현 그룹 → 종양 비율 히트맵</h3>'
html += img(f'{D}04_Interaction_Heatmap_BRCA1.png')

html += """<div class="insight"><strong>핵심 인사이트</strong>: TSS(수집 기관)에 따라 BRCA1/BRCA2 발현 패턴이 다르며,
이는 환자 모집 기관별 코호트 특성 차이를 반영합니다. 특히 BH, E2 기관은 정상 대조군이 있어
종양/정상 비교가 가능하지만, A2, A8 등은 종양만 수집되어 기관 효과 통제가 필요합니다.</div>"""

# === 5 ===
html += '<h2 id="s5" class="page-break">5. 교차분석 &amp; 카이제곱 검정</h2>'
html += tbl(chi)

html += '<h3>TSS Code별 종양/정상 분포 (상위 10개 기관)</h3>'
tss_top = tss.head(10).copy()
tss_top.columns = ['TSS Code', '종양 수', '전체', '정상 수', '종양 비율(%)']
html += tbl(tss_top)

html += """<div class="insight"><strong>핵심 인사이트</strong>: Cramer's V = <strong>0.435</strong>로 TSS와 종양 여부 간
<strong>중간~강한 연관성</strong>이 있습니다. BH(67.8%), E2(89.4%)만 정상 조직이 있고,
A2, A8, D8 등 다수 기관은 종양만 100% 수집했습니다.
이는 기관 간 불균형을 의미하므로, 모델링 시 주의가 필요합니다.</div>"""

# === 6 ===
html += '<h2 id="s6" class="page-break">6. 분포 시각화</h2>'

html += '<h3>6-1. BRCA1, BRCA2, ESR1, PGR 분포 및 Boxplot</h3>'
html += img(f'{D}06_BRCA_Distributions_Boxplots.png')

html += """<p>상단: Log2(TPM+1) 분포 (빨간 점선=평균, 파란 실선=중앙값)<br>
하단: 종양(1) vs 정상(0) Boxplot</p>"""

html += '<h3>6-2. BRCA1 vs BRCA2 산점도</h3>'
html += img(f'{D}06_Scatter_BRCA1_vs_BRCA2.png')

html += '<h3>6-3. ESR1 vs PGR 산점도 (호르몬 수용체)</h3>'
html += img(f'{D}06_Scatter_ESR1_vs_PGR.png')

html += """<div class="insight"><strong>핵심 인사이트</strong>:
<ul style="margin:0.3em 0;">
<li>BRCA1과 BRCA2는 강한 양의 상관을 보이며, 종양(빨간색)에서 두 유전자 모두 높은 발현</li>
<li>ESR1-PGR 산점도에서 <strong>두 클러스터</strong>가 관찰됨: ER+/PR+ 그룹(우상단)과 ER-/PR- 그룹(좌하단)
— 이는 유방암의 호르몬 수용체 서브타입을 반영</li>
</ul></div>"""

# === 7 ===
html += '<h2 id="s7" class="page-break">7. 전수조사 결과 (Top 20 유전자)</h2>'
html += """<p>23,368개 전체 유전자를 <strong>BRCA 환자 1,218명</strong>에 대해서만 순회하며
|Point-Biserial 상관계수|가 가장 큰 20개 유전자를 자동 발굴했습니다.</p>"""

top20_show = top20[['Gene','Mean_Normal','Mean_Tumor','Log2FC','P_Value','PB_Corr','Abs_PB_Corr']].copy()
top20_show.columns = ['유전자','평균(정상)','평균(종양)','Log2FC','p-value','PB상관계수','|PB상관계수|']
html += tbl(top20_show)

html += '<h3>Top 20 유전자 시각화</h3>'
html += img(f'{D}08_BRCA_Top20_BarChart.png')

html += """<div class="insight"><strong>핵심 인사이트</strong>: 유방암 환자에서 종양과 가장 강한 상관을 보이는 유전자 Top 5:
<ol style="margin:0.3em 0;">
<li><strong>FIGF</strong> (r=-0.863): VEGF-D로도 불리며, 정상 유방 조직에서 고발현 → 종양에서 소실</li>
<li><strong>CA4</strong> (r=-0.789): Carbonic Anhydrase 4, 종양에서 극적 감소 (54.5→1.4 TPM)</li>
<li><strong>TSLP</strong> (r=-0.775): 면역 사이토카인, 종양 미세환경 변화 반영</li>
<li><strong>CD300LG</strong> (r=-0.763): 면역세포 마커, 종양에서 67.9→2.2 TPM으로 소실</li>
<li><strong>SCARA5</strong> (r=-0.752): Scavenger Receptor, 종양 억제 기능</li>
</ol>
<strong>모두 음의 상관</strong> → 유방암 종양에서 정상 대비 발현이 극적으로 감소하는 유전자들입니다.
이들은 종양 미세환경의 파괴, 면역 회피, 혈관 리모델링을 반영합니다.</div>"""

html += '<h3>Volcano Plot</h3>'
html += img(f'{D}07_BRCA_Volcano_Plot.png')
html += '<p>회색: 전체 유전자, 빨간색: |Log2FC|>1 & p<0.05. BRCA1/BRCA2 및 주요 암 유전자 위치가 표시되어 있습니다.</p>'

# === 8 ===
html += '<h2 id="s8" class="page-break">8. BRCA1/BRCA2 심층 분석</h2>'

html += '<h3>8-1. BRCA1/BRCA2 전수조사 결과</h3>'
bd = brca_det[['Gene','Mean_Normal','Mean_Tumor','Log2FC','T_Stat','P_Value','PB_Corr']].copy()
bd.columns = ['유전자','평균(정상)','평균(종양)','Log2FC','T-통계량','p-value','PB상관계수']
html += tbl(bd)

html += '<h3>8-2. BRCA1 발현 사분위 분석</h3>'
q = quartile.copy()
q.columns = ['BRCA1 그룹','샘플 수','종양 비율(%)','BRCA1 평균','BRCA2 평균']
html += tbl(q)

html += img(f'{D}09_BRCA1_Quartile_Plot.png')

html += """<div class="insight"><strong>핵심 인사이트</strong>:
<ul style="margin:0.3em 0;">
<li>BRCA1 발현이 높을수록 종양 비율이 급증: Q1(Low) <strong>84.3%</strong> → Q4(High) <strong>99.7%</strong></li>
<li>BRCA1과 BRCA2는 동조적으로 증가: BRCA1 Q4에서 BRCA2도 4.4 TPM (Q1의 2.8배)</li>
<li>이는 <strong>DNA 손상 복구(DDR) 경로의 전반적 활성화</strong>를 시사 — 유방암 종양에서
게놈 불안정성에 대한 보상적 반응으로 해석됩니다.</li>
</ul></div>"""

# === 9 ===
html += '<h2 id="s9" class="page-break">9. BRCA vs 다른 암종 비교</h2>'

html += '<p>BRCA1/BRCA2 등 핵심 유전자의 발현량을 유방암 종양과 다른 23개 암종 종양 간 비교했습니다.</p>'
vo = vs_other.copy()
vo.columns = ['유전자','유방암 평균','타 암종 평균','비율','T-통계량','p-value']
html += tbl(vo)

html += img(f'{D}10_BRCA_vs_OtherCancers.png')

html += """<div class="insight"><strong>핵심 인사이트</strong>:
<ul style="margin:0.3em 0;">
<li><strong>ESR1</strong>: 유방암에서 타 암종 대비 <strong>17.0배</strong> 높은 발현 — 유방암 특이적 마커</li>
<li><strong>PGR</strong>: 유방암에서 <strong>10.9배</strong> 높은 발현 — 호르몬 수용체 양성 유방암의 특징</li>
<li><strong>BRCA1</strong>(1.16배), <strong>BRCA2</strong>(1.23배): 유방암에서 약간 높지만 차이가 크지 않음
→ BRCA1/2는 유방암 특이적이라기보다 <strong>범암종적(pan-cancer)</strong> DDR 마커</li>
<li><strong>TP53</strong>(1.10배): 암종 간 차이 미미</li>
</ul></div>"""

# === 10 ===
html += '<h2 id="s10" class="page-break">10. 종합 요약 및 시사점</h2>'
html += """
<h3>주요 발견사항</h3>
<ol>
<li><strong>BRCA 환자 1,218명 추출 성공</strong>: TSS 코드 기반으로 종양 1,105명 + 정상 113명 식별</li>
<li><strong>BRCA1/BRCA2 과발현 확인</strong>: 종양에서 각각 2.0배, 2.6배 증가 (p≈0)</li>
<li><strong>유방암 내 종양 구분력</strong>: EGFR (r=-0.477)이 가장 강한 단일 피처, 이어서 MYC, PTEN, BRCA2 순</li>
<li><strong>전수조사 Top 유전자</strong>: FIGF, CA4, TSLP 등 종양 미세환경 관련 유전자가 |r|>0.7로 지배적</li>
<li><strong>ESR1/PGR</strong>: 유방암 특이적 발현 (타 암종 대비 10~17배) — 호르몬 수용체 서브타입 구분에 활용 가능</li>
<li><strong>TP53</strong>: 유방암 내에서 종양/정상 구분력 없음 (p=0.49) — 발현량보다 돌연변이 여부가 중요</li>
<li><strong>BRCA1 발현 사분위</strong>: Q4(고발현) 그룹의 99.7%가 종양 — BRCA1 발현량 자체가 유방암 바이오마커</li>
</ol>

<h3>향후 연구를 위한 제안</h3>
<table>
<thead><tr><th>항목</th><th>제안</th></tr></thead>
<tbody>
<tr><td>타겟 변수</td><td>Is_Tumor (종양/정상 분류) 또는 호르몬 수용체 서브타입 분류</td></tr>
<tr><td>피처 선택</td><td>전수조사 Top 20 + BRCA1, BRCA2, ESR1, PGR</td></tr>
<tr><td>서브타입 분석</td><td>ESR1/PGR 발현 기반 ER+/PR+ vs Triple Negative 분류</td></tr>
<tr><td>전처리</td><td>Log2(TPM+1) 변환, PIK3CA/EGFR 이상치 처리, TSS 기관 효과 보정</td></tr>
<tr><td>TDA 적용</td><td>BRCA1/BRCA2 + Top 20 유전자로 유방암 환자의 위상학적 구조 탐색</td></tr>
</tbody>
</table>

<h3 style="margin-top:2em;">부록: 생성 파일 목록</h3>
<table>
<thead><tr><th>파일명</th><th>유형</th><th>설명</th></tr></thead>
<tbody>
<tr><td>00_BRCA_All_23368_Genes_Statistics.csv</td><td>CSV</td><td>BRCA 환자 전수조사 통계 (23,368개)</td></tr>
<tr><td>02_BRCA_TTest_Results.csv</td><td>CSV</td><td>주요 유전자 T-검정</td></tr>
<tr><td>03_BRCA_Point_Biserial.csv</td><td>CSV</td><td>Point-Biserial 상관</td></tr>
<tr><td>07_BRCA_Volcano_Plot.png</td><td>이미지</td><td>Volcano Plot</td></tr>
<tr><td>08_BRCA_Top20_Genes.csv</td><td>CSV</td><td>전수조사 Top 20 유전자</td></tr>
<tr><td>09_BRCA1_BRCA2_Detail.csv</td><td>CSV</td><td>BRCA1/BRCA2 상세 통계</td></tr>
<tr><td>09_BRCA1_Quartile_Analysis.csv</td><td>CSV</td><td>BRCA1 사분위 분석</td></tr>
<tr><td>10_BRCA_vs_OtherCancers.csv</td><td>CSV</td><td>유방암 vs 타 암종 비교</td></tr>
</tbody>
</table>
"""

html += '</body></html>'

# Save HTML
html_path = 'TCGA_BRCA_Patient_Analysis_Report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"HTML 생성 완료: {html_path}")

# Generate PDF
pdf_path = 'TCGA_BRCA_Patient_Analysis_Report.pdf'
weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
print(f"PDF 생성 완료: {pdf_path}")

print("Done!")
