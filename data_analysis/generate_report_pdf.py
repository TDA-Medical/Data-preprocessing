import pandas as pd
import numpy as np
import os
import base64
import weasyprint

OUTPUT_DIR_ORIG = './output/'
OUTPUT_DIR_FULL = './output_full_genes/'
OUTPUT_DIR_BAD = './output_bad_genes/'

def img_to_base64(path):
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def df_to_html_table(df, bold_cols=None):
    """Convert DataFrame to styled HTML table."""
    html = '<table>\n<thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead>\n<tbody>\n'
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                if abs(val) < 0.001 and val != 0:
                    val = f'{val:.2e}'
                elif abs(val) > 100:
                    val = f'{val:,.1f}'
                else:
                    val = f'{val:.4f}'
            bold = bold_cols and col in bold_cols
            td = f'<td><strong>{val}</strong></td>' if bold else f'<td>{val}</td>'
            html += td
        html += '</tr>\n'
    html += '</tbody></table>'
    return html

def img_tag(path, width='100%'):
    b64 = img_to_base64(path)
    if not b64:
        return '<p><em>(이미지 없음)</em></p>'
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};">'

# ---------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------
overview = pd.read_csv(f'{OUTPUT_DIR_ORIG}01_Data_Overview.csv')
desc_stats = pd.read_csv(f'{OUTPUT_DIR_ORIG}02_Numeric_Descriptive_Stats.csv')
ttest = pd.read_csv(f'{OUTPUT_DIR_ORIG}02_T_Test_Results.csv')
pb_corr = pd.read_csv(f'{OUTPUT_DIR_ORIG}03_Point_Biserial_Corr.csv')
chi_orig = pd.read_csv(f'{OUTPUT_DIR_ORIG}05_ChiSquare_Results.csv')

fullscan_stats = pd.read_csv(f'{OUTPUT_DIR_FULL}00_All_23368_Genes_Statistics.csv')
chi_full = pd.read_csv(f'{OUTPUT_DIR_FULL}05_ChiSquare_Results.csv')

venn = pd.read_csv(f'{OUTPUT_DIR_BAD}01_Venn_Summary.csv')
bad_comp = pd.read_csv(f'{OUTPUT_DIR_BAD}02_Bad_vs_NotBad_Comparison.csv')
brca_detail = pd.read_csv(f'{OUTPUT_DIR_BAD}08_BRCA_Genes_Detail.csv')
gene_family = pd.read_csv(f'{OUTPUT_DIR_BAD}09_Gene_Family_Enrichment.csv')

# Fullscan top genes
fullscan_stats['Abs_Corr'] = fullscan_stats['Point_Biserial_Corr'].abs()
top5_full = fullscan_stats.nlargest(5, 'Abs_Corr')
top20_full = fullscan_stats.nlargest(20, 'Abs_Corr')

# Counts
n_tumor = overview[overview['Metric']=='Total Samples']['Value'].values[0]
target_ratio = overview[overview['Metric']=='Target Ratio (Tumor : Normal)']['Value'].values[0]

# ---------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------
html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
@page {{
    size: A4;
    margin: 2cm 2.5cm;
}}
body {{
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    font-size: 10.5pt;
    line-height: 1.7;
    color: #222;
}}
h1 {{
    font-size: 22pt;
    font-weight: 900;
    margin-bottom: 0.3em;
    color: #111;
}}
h2 {{
    font-size: 16pt;
    font-weight: 800;
    margin-top: 1.8em;
    margin-bottom: 0.5em;
    color: #111;
    page-break-after: avoid;
}}
h3 {{
    font-size: 13pt;
    font-weight: 700;
    margin-top: 1.3em;
    margin-bottom: 0.4em;
    color: #222;
    page-break-after: avoid;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 0.8em 0 1.2em 0;
    font-size: 9.5pt;
}}
th {{
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 6px 10px;
    text-align: center;
    font-weight: 700;
}}
td {{
    border: 1px solid #ddd;
    padding: 5px 10px;
    text-align: center;
}}
tr:nth-child(even) {{
    background: #fafafa;
}}
.info-box {{
    background: #f8f9fa;
    border-left: 4px solid #4a90d9;
    padding: 12px 16px;
    margin: 1em 0;
    font-size: 10pt;
}}
.insight-box {{
    background: #fff8e6;
    border-left: 4px solid #e6a817;
    padding: 10px 14px;
    margin: 0.8em 0;
    font-size: 9.5pt;
}}
.toc {{
    margin: 1.5em 0;
}}
.toc ol {{
    line-height: 2.2;
}}
.toc a {{
    color: #0077cc;
    text-decoration: none;
}}
img {{
    max-width: 100%;
    display: block;
    margin: 0.5em auto;
}}
.page-break {{
    page-break-before: always;
}}
</style>
</head>
<body>

<h1>TCGA RNA-Seq 통계 분석 보고서</h1>

<div class="info-box">
<strong>분석 일자</strong>: 2026-03-18<br>
<strong>데이터셋</strong>: GSE62944 (TCGA RNA-Seq TPM, Rsubread 정렬)<br>
<strong>분석 대상</strong>: {n_tumor}개 샘플 / 23,368개 유전자 전수조사<br>
<strong>타겟 비율</strong>: Tumor : Normal = {target_ratio}
</div>

<div class="toc">
<h2>목차</h2>
<ol>
<li><a href="#sec1">데이터 개요</a></li>
<li><a href="#sec2">기술통계</a></li>
<li><a href="#sec3">상관분석</a></li>
<li><a href="#sec4">교호작용 분석</a></li>
<li><a href="#sec5">교차분석 &amp; 카이제곱 검정</a></li>
<li><a href="#sec6">분포 시각화</a></li>
<li><a href="#sec7">전수조사 결과 (23,368개 유전자)</a></li>
<li><a href="#sec8">Bad Genes 분석</a></li>
<li><a href="#sec9">BRCA 유전자 심층 분석</a></li>
<li><a href="#sec10">종합 요약 및 시사점</a></li>
</ol>
</div>

<!-- ============================================================ -->
<h2 id="sec1" class="page-break">1. 데이터 개요</h2>

<p>GSE62944는 TCGA(The Cancer Genome Atlas) RNA-Seq 데이터를 Rsubread로 재정렬한 것으로,
24개 암종(cancer type)에 걸친 종양(Tumor)과 정상(Normal) 조직의 유전자 발현량(TPM)을 포함합니다.</p>

<h3>분석에 사용된 핵심 데이터</h3>

<table>
<thead><tr><th>항목</th><th>값</th><th>설명</th></tr></thead>
<tbody>
<tr><td>종양 샘플 수</td><td>9,264</td><td>24개 암종 포함</td></tr>
<tr><td>정상 샘플 수</td><td>741</td><td>정상 조직 대조군</td></tr>
<tr><td>전체 샘플 수</td><td>{n_tumor}</td><td>종양 + 정상</td></tr>
<tr><td>유전자 수</td><td>23,368</td><td>전수조사 완료</td></tr>
<tr><td>타겟 비율</td><td>{target_ratio}</td><td>심한 클래스 불균형</td></tr>
</tbody>
</table>

<div class="insight-box">
<strong>핵심 인사이트</strong>: 종양 대 정상 비율이 약 <strong>12.5:1</strong>로 심한 클래스 불균형을 보입니다.
분류 모델링 시 oversampling 또는 loss weighting이 필수적입니다.
</div>

<!-- ============================================================ -->
<h2 id="sec2" class="page-break">2. 기술통계</h2>

<h3>2-1. 핵심 유전자 5종 기술통계 요약</h3>
"""

# Descriptive stats table
desc = desc_stats.copy()
desc = desc.rename(columns={'Unnamed: 0': '유전자'})
cols_show = ['유전자', 'count', 'mean', 'std', '1%', '25%', '50%', '75%', '99%', 'max', 'Skewness', 'Kurtosis']
desc_show = desc[[c for c in cols_show if c in desc.columns]]
html += df_to_html_table(desc_show)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: EGFR은 왜도 13.96, 첨도 283.4로 <strong>극심한 우편향</strong>을 보이며,
모델링 시 <strong>로그 변환</strong>이나 <strong>이상치 처리</strong>가 필수적입니다.
모든 유전자가 양의 왜도를 보여 TPM 분포의 비대칭성을 확인할 수 있습니다.
</div>
"""

# T-test results
html += '<h3>2-2. Tumor vs Normal T-검정 결과</h3>'
html += df_to_html_table(ttest)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: BRCA1은 종양에서 정상 대비 <strong>2.6배</strong> 높은 발현량을 보이며
(t = -34.40, p ≈ 0), 5개 유전자 중 가장 강력한 차별적 발현을 보입니다.
PTEN은 종양에서 오히려 <strong>감소</strong>하여 종양 억제 유전자 역할을 확인합니다.
</div>
"""

# ============================================================
html += '<h2 id="sec3" class="page-break">3. 상관분석</h2>'

html += '<h3>3-1. Pearson &amp; Spearman 상관계수 히트맵</h3>'
html += img_tag(f'{OUTPUT_DIR_ORIG}03_Correlation_Heatmaps.png')

html += '<h3>3-2. Point-Biserial 상관 (종양 여부 vs 유전자 발현)</h3>'
html += df_to_html_table(pb_corr)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: 종양 예측에 가장 강한 단일 변수는 <strong>BRCA1</strong> (r_pb = 0.169)이며,
PTEN은 음의 상관 (r_pb = -0.109)으로 종양에서 하향 조절됨을 보여줍니다.
MYC (r_pb = 0.023)는 의외로 약한 상관을 보여, 단독 바이오마커로는 부적합합니다.
</div>
"""

# ============================================================
html += '<h2 id="sec4" class="page-break">4. 교호작용 분석</h2>'

html += '<h3>4-1. TSS Code x 종양 여부 -> 유전자 발현량</h3>'
# Original interaction plots
html += img_tag(f'{OUTPUT_DIR_ORIG}04_Interaction_Cat_Cat.png')
html += '<p>TSS Code(조직 기원)와 샘플 유형(Sample Type)의 교차 종양 비율을 히트맵으로 시각화한 것입니다.</p>'

html += '<h3>4-2. TSS Code x TP53 발현 그룹 -> 종양 비율</h3>'
html += img_tag(f'{OUTPUT_DIR_ORIG}04_Interaction_Cat_NumGroup.png')

html += '<h3>4-3. TSS Code x EGFR 발현 교호작용</h3>'
html += img_tag(f'{OUTPUT_DIR_ORIG}04_Interaction_TSS_EGFR.png')

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: TSS Code(조직 기원 기관)에 따라 유전자 발현 패턴이 크게 달라지며,
이는 암종(cancer type)별 이질성을 반영합니다.
특히 EGFR는 TSS별로 종양/정상 간 발현 차이의 방향성이 반전되는 경우도 관찰됩니다.
</div>
"""

# ============================================================
html += '<h2 id="sec5" class="page-break">5. 교차분석 &amp; 카이제곱 검정</h2>'

html += df_to_html_table(chi_orig)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: TSS_Code와 종양 여부 간 Cramer's V = <strong>0.458</strong>로 중간~강한 연관성을 보입니다.
이는 특정 기관/조직에서 종양 샘플이 집중적으로 수집되었음을 의미하며,
분석 시 기관 효과(site effect)를 통제할 필요가 있습니다.
</div>
"""

# ============================================================
html += '<h2 id="sec6" class="page-break">6. 분포 시각화</h2>'

html += '<h3>6-1. 주요 유전자 Log2(TPM+1) 분포</h3>'
html += img_tag(f'{OUTPUT_DIR_ORIG}06_Distributions_Histograms_Log.png')
html += '<p>빨간 점선은 평균, 파란 실선은 중앙값입니다. 평균과 중앙값의 차이가 클수록 분포가 비대칭입니다.</p>'

html += '<h3>6-2. 종양 여부별 Boxplot</h3>'
html += img_tag(f'{OUTPUT_DIR_ORIG}06_Boxplots_by_Target.png')

html += '<h3>6-3. EGFR vs MYC 산점도 (종양 색상 구분)</h3>'
html += img_tag(f'{OUTPUT_DIR_ORIG}06_Scatter_EGFR_MYC.png')
html += '<p>투명도를 0.3으로 조절하여 고밀도 영역을 식별할 수 있습니다.</p>'

# ============================================================
html += '<h2 id="sec7" class="page-break">7. 전수조사 결과 (23,368개 유전자)</h2>'

html += """<p>23,368개 전체 유전자를 메모리 스트리밍 방식으로 순회하며 T-검정과 Point-Biserial 상관계수를
계산했습니다. 아래는 <strong>데이터 주도적(Data-Driven)</strong>으로 발굴된 상위 20개 핵심 유전자입니다.</p>"""

html += '<h3>7-1. |Point-Biserial 상관계수| 상위 20개 유전자</h3>'
top20_show = top20_full[['Gene', 'Mean_Normal', 'Mean_Tumor', 'Log2_T_Stat', 'Log2_T_PValue', 'Point_Biserial_Corr', 'Abs_Corr']].copy()
top20_show.columns = ['유전자', '평균(정상)', '평균(종양)', 'T-통계량(Log2)', 'p-value', 'PB 상관계수', '|PB 상관계수|']
html += df_to_html_table(top20_show)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: 전수조사 결과, 종양과 가장 강한 상관을 보이는 유전자는 <strong>CCL14, SCARA5, TNXB, FIGF, CD300LG</strong>입니다.
이들은 모두 음의 상관계수를 보여, <strong>종양에서 발현이 억제되는 유전자</strong>들입니다.
주로 면역 반응(CCL14), 세포외기질(TNXB), 혈관신생(FIGF) 관련 유전자로,
종양 미세환경(TME)의 변화를 반영합니다.
</div>
"""

html += '<h3>7-2. 전수조사 기반 시각화</h3>'
html += '<h4>분포 및 Boxplot</h4>'
html += img_tag(f'{OUTPUT_DIR_FULL}06_Distributions_and_Boxplots.png')

html += '<h4>상위 2개 유전자 산점도</h4>'
html += img_tag(f'{OUTPUT_DIR_FULL}06_Scatter_Top2_Genes.png')

html += '<h4>교호작용 히트맵</h4>'
html += img_tag(f'{OUTPUT_DIR_FULL}04_Interaction_Heatmap.png')

# ============================================================
html += '<h2 id="sec8" class="page-break">8. Bad Genes 분석</h2>'

html += """<p>논문에서 제공된 "Bad Genes" 목록(TCGA 1,637개, Overall 2,068개)을 전체 유전자 발현 데이터와
교차 분석하여, 이들 유전자의 통계적 특성을 비교했습니다.</p>"""

html += '<h3>8-1. Bad Genes 목록 Venn 요약</h3>'
venn_show = venn.copy()
venn_show.columns = ['분류', '목록 수', '데이터 매칭 수']
html += df_to_html_table(venn_show)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: TCGA bad genes 1,637개는 Overall bad genes에 <strong>100% 포함</strong>되어 있습니다 (부분집합).
Overall 목록에는 추가로 431개 유전자가 있으며, 이 중 383개가 GSE62944 데이터에서 매칭됩니다.
</div>
"""

html += '<h3>8-2. Bad vs Not-Bad 유전자 비교</h3>'
bad_show = bad_comp.copy()
bad_show.columns = ['지표', 'Bad Genes', 'Not-Bad Genes']
html += df_to_html_table(bad_show)

html += '<h3>8-3. Bad vs Not-Bad 분포 비교</h3>'
html += img_tag(f'{OUTPUT_DIR_BAD}03_Bad_vs_NotBad_Distributions.png')

html += '<h3>8-4. Bad Gene 카테고리별 비교</h3>'
html += img_tag(f'{OUTPUT_DIR_BAD}04_Bad_Category_Comparison.png')

html += '<h3>8-5. Volcano Plot (Bad Genes 하이라이트)</h3>'
html += img_tag(f'{OUTPUT_DIR_BAD}05_Volcano_Bad_Genes.png')
html += '<p>회색: Not-Bad, 파란색: TCGA Only, 초록색: Overall Only, 빨간색: Both. BRCA1/BRCA2 위치가 표시되어 있습니다.</p>'

html += '<h3>8-6. 상위 20개 유의미한 Bad Genes</h3>'
html += img_tag(f'{OUTPUT_DIR_BAD}07_Top20_Bad_Genes.png')

html += '<h3>8-7. 유전자 패밀리별 Bad Gene 비율</h3>'
gf_show = gene_family.copy()
gf_show.columns = ['유전자 패밀리', 'Bad 수', '전체 수', 'Bad 비율(%)']
html += df_to_html_table(gf_show)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>: HLA(면역) 유전자의 <strong>66.7%</strong>가 bad gene 목록에 포함되어 있어 가장 높은 비율을 보입니다.
리보솜 단백질(RPL 54.1%, RPS 44.6%)도 높은 비율로, 이는 하우스키핑 유전자의 특성상
조직 간 발현 변이가 크기 때문으로 해석됩니다.
</div>
"""

# ============================================================
html += '<h2 id="sec9" class="page-break">9. BRCA 유전자 심층 분석</h2>'

brca_show = brca_detail[['Gene', 'Mean_Normal', 'Mean_Tumor', 'Log2FC', 'T_Stat', 'P_Value', 'PB_Corr', 'Bad_Category']].copy()
brca_show.columns = ['유전자', '평균(정상)', '평균(종양)', 'Log2FC', 'T-통계량', 'p-value', 'PB 상관계수', 'Bad 분류']
html += df_to_html_table(brca_show)

html += """
<div class="insight-box">
<strong>핵심 인사이트</strong>:
<ul style="margin:0.3em 0;">
<li><strong>BRCA1</strong>: 종양에서 정상 대비 <strong>2.6배</strong> 높은 발현 (Log2FC = 0.91, p ≈ 0). DNA 손상 복구 경로의 활성화를 반영합니다.</li>
<li><strong>BRCA2</strong>: 종양에서 정상 대비 <strong>3.6배</strong> 높은 발현 (Log2FC = 0.72, p ≈ 0). BRCA1과 유사한 패턴입니다.</li>
<li>두 유전자 모두 Bad Gene 목록에 <strong>포함되지 않아</strong>, 종양 연구에서 신뢰할 수 있는 바이오마커입니다.</li>
</ul>
</div>
"""

# ============================================================
html += '<h2 id="sec10" class="page-break">10. 종합 요약 및 시사점</h2>'

html += """
<h3>주요 발견사항</h3>
<ol>
<li><strong>클래스 불균형</strong>: 종양 92.6% vs 정상 7.4%로 심한 불균형 — 모델링 시 oversampling/loss weighting 필요</li>
<li><strong>BRCA1이 가장 강력한 차별적 발현 유전자</strong>: 핵심 5개 유전자 중 t-통계량 절대값이 가장 크고 (34.40), PB 상관계수도 가장 높음 (0.169)</li>
<li><strong>전수조사 결과 종양 미세환경(TME) 유전자가 지배적</strong>: CCL14, SCARA5 등 면역/기질 관련 유전자가 종양과 가장 강한 상관을 보임</li>
<li><strong>Bad Genes의 통계적 특성</strong>: Bad gene 목록의 유전자들이 Not-Bad 대비 약간 높은 |PB_Corr|과 |Log2FC|를 보이나, 차이는 크지 않음</li>
<li><strong>HLA 유전자 67%가 Bad Gene</strong>: 면역 관련 유전자의 발현 변이가 커서 분석 시 주의 필요</li>
<li><strong>BRCA1/BRCA2는 Bad Gene 아님</strong>: 안정적인 바이오마커로 활용 가능</li>
</ol>

<h3>향후 연구를 위한 제안</h3>
<table>
<thead><tr><th>항목</th><th>제안</th></tr></thead>
<tbody>
<tr><td>타겟 변수</td><td>Is_Tumor (종양 여부 분류) 또는 특정 암종 분류</td></tr>
<tr><td>피처 선택</td><td>전수조사 Top 20 유전자 + BRCA1/BRCA2 포함</td></tr>
<tr><td>Bad Gene 처리</td><td>HLA, 리보솜 유전자 등 Bad Gene 목록 참고하여 노이즈 피처 제거 고려</td></tr>
<tr><td>전처리</td><td>Log2(TPM+1) 변환, 클래스 불균형 처리, TSS 기관 효과 통제</td></tr>
<tr><td>TDA 적용</td><td>전수조사로 선별된 핵심 유전자를 피처로 사용하여 위상학적 분석 수행</td></tr>
</tbody>
</table>

<h3 style="margin-top:2em;">부록: 생성 파일 목록</h3>
<table>
<thead><tr><th>파일명</th><th>유형</th><th>설명</th></tr></thead>
<tbody>
<tr><td>output/01_Data_Overview.csv</td><td>CSV</td><td>데이터 개요</td></tr>
<tr><td>output/02_Numeric_Descriptive_Stats.csv</td><td>CSV</td><td>수치형 변수 기술통계</td></tr>
<tr><td>output/02_T_Test_Results.csv</td><td>CSV</td><td>T-검정 결과</td></tr>
<tr><td>output/03_Correlation_Heatmaps.png</td><td>이미지</td><td>상관계수 히트맵</td></tr>
<tr><td>output/03_Point_Biserial_Corr.csv</td><td>CSV</td><td>Point-Biserial 상관</td></tr>
<tr><td>output_full_genes/00_All_23368_Genes_Statistics.csv</td><td>CSV</td><td>전수조사 통계 (23,368개)</td></tr>
<tr><td>output_bad_genes/05_Volcano_Bad_Genes.png</td><td>이미지</td><td>Bad Gene Volcano Plot</td></tr>
<tr><td>output_bad_genes/06_All_Bad_Genes_Ranked.csv</td><td>CSV</td><td>Bad Gene 유의성 순위</td></tr>
<tr><td>output_bad_genes/08_BRCA_Genes_Detail.csv</td><td>CSV</td><td>BRCA 유전자 상세</td></tr>
</tbody>
</table>
"""

html += '</body></html>'

# ---------------------------------------------------------------
# Generate PDF
# ---------------------------------------------------------------
print("HTML 생성 완료. PDF 변환 중...")

html_path = 'TCGA_RNASeq_Statistical_Analysis_Report.html'
pdf_path = 'TCGA_RNASeq_Statistical_Analysis_Report.pdf'

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)

weasyprint.HTML(filename=html_path).write_pdf(pdf_path)
print(f"PDF 보고서 생성 완료: {pdf_path}")
