=======
Archive
=======


The 2014 archive is the public OptBench site at the commit named in
the oracle.
Chill, Stevenson, Ruehle, Shang, Xiao, Farrell, Wales, and Henkelman
set those problems in J. Chem. Theory Comput. 2014, 10, 5476-5482
(`doi:10.1021/ct5008718 <https://doi.org/10.1021/ct5008718>`_).
A run in that archive counts as finished when it stops under 10000
force calls.

Each problem below leads with the best full harness run.

.. raw:: html

   <section class="bench" id="collection">
   <p>Each problem leads with the best full harness run.</p>
   <h2 id="suite-minimization">Minimization</h2>
   <h3>Lennard-Jones 38 clusters</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="minimization/lj38">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">rgmin/lj38-dogleg</td><td data-field="version">rgmin 0.4.1</td><td data-field="commit">9a2e7ff11fa6</td><td data-field="source">campaign, asv</td><td data-field="par2_mean">81.881</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.996173</td><td data-field="wilson_high">1</td><td data-field="median_calls">79</td><td data-field="mean_calls">81.881</td><td data-field="archive"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">1000</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">Files with no per-start harness record stay off the ranked table.</p>
   <p><a href="generated/suites/minimization/lj38.html#lj38">Full ranked table and performance profile</a>.</p>
   <h3>Lennard-Jones 38 clusters with Gaussian force noise</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="minimization/lj38-noise">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">noise/rgmin-dogleg-s0</td><td data-field="version">rgmin 0.4.1</td><td data-field="commit">9a2e7ff11fa6</td><td data-field="source">campaign, asv</td><td data-field="par2_mean">81.881</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.996173</td><td data-field="wilson_high">1</td><td data-field="median_calls">79</td><td data-field="mean_calls">81.881</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">1000</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/minimization/lj38-noise.html#lj38-noise">Full ranked table and performance profile</a>.</p>
   <h3>Morse bulk</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="minimization/morse-bulk">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">rgmin/morse-newton</td><td data-field="version">rgmin 0.4.1</td><td data-field="commit">9a2e7ff11fa6</td><td data-field="source">campaign, asv</td><td data-field="par2_mean">8.35</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.963007</td><td data-field="wilson_high">1</td><td data-field="median_calls">9</td><td data-field="mean_calls">8.35</td><td data-field="archive"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">100</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">Files with no per-start harness record stay off the ranked table.</p>
   <p><a href="generated/suites/minimization/morse-bulk.html#morse-bulk">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-saddle-search">Saddle search</h2>
   <h3>Pt heptamer island</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="saddle-search/pt-island">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Runs</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">sella/pt-island-search-default</td><td data-field="version">Sella 2.6.0</td><td data-field="commit">6cfb60cdb100</td><td data-field="source">asv</td><td data-field="par2_mean">5201.62</td><td data-field="success_rate">0.75</td><td data-field="wilson_low">0.505017</td><td data-field="wilson_high">0.898179</td><td data-field="median_calls">225</td><td data-field="mean_calls">247.812</td><td data-field="archive"></td><td data-field="runs"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 4</td><td data-field="n">16</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">Eon rows are not yet on the harness counter.</p>
   <p><a href="generated/suites/saddle-search/pt-island.html#pt-island">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-double-ended-saddle-search">Double-ended saddle search</h2>
   <h3>LJ38 double-ended saddle</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="double-ended-saddle-search/lj38">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">eon-svn/de-lj38-lanczos-improved</td><td data-field="version">Eon svn r2549</td><td data-field="commit">r2549</td><td data-field="source">campaign</td><td data-field="par2_mean">235.2</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.928652</td><td data-field="wilson_high">1</td><td data-field="median_calls">197</td><td data-field="mean_calls">235.2</td><td data-field="mean_per_image">47.04</td><td data-field="archive"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">50</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">eOn band rows are not yet on the harness counter. Files with no per-start harness record stay off the ranked table.</p>
   <p><a href="generated/suites/double-ended-saddle-search/lj38.html#lj38">Full ranked table and performance profile</a>.</p>
   <h3>lj38_2ts_separation</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="double-ended-saddle-search/lj38_2ts_separation">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">eon-svn/de-2ts-lanczos-improved</td><td data-field="version">Eon svn r2549</td><td data-field="commit">r2549</td><td data-field="source">campaign</td><td data-field="par2_mean">502.784</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.929953</td><td data-field="wilson_high">1</td><td data-field="median_calls">418</td><td data-field="mean_calls">502.784</td><td data-field="mean_per_image">100.557</td><td data-field="archive"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">51</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/double-ended-saddle-search/lj38_2ts_separation.html#lj38_2ts_separation">Full ranked table and performance profile</a>.</p>
   <h3>pt-island</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="double-ended-saddle-search/pt-island">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">eon-svn/de-pt-lanczos-improved</td><td data-field="version">Eon svn r2549</td><td data-field="commit">r2549</td><td data-field="source">campaign</td><td data-field="par2_mean">180.644</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.938871</td><td data-field="wilson_high">1</td><td data-field="median_calls">161</td><td data-field="mean_calls">180.644</td><td data-field="mean_per_image">60.2147</td><td data-field="archive">eon-lanczos-improved: Eon r2156, force calls 222</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">59</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">eOn band rows are not yet on the harness counter.</p>
   <p><a href="generated/suites/double-ended-saddle-search/pt-island.html#pt-island">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-globalopt">Global optimization</h2>
   <h3>Binary Lennard-Jones 100</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj100-binary">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/lj100-binary.html#lj100-binary">Full ranked table and performance profile</a>.</p>
   <h3>Lennard-Jones 38 global minimum</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj38">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/lj38.html#lj38">Full ranked table and performance profile</a>.</p>
   <h3>lj150</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj150">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/lj150.html#lj150">Full ranked table and performance profile</a>.</p>
   <h3>lj75</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj75">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/lj75.html#lj75">Full ranked table and performance profile</a>.</p>
   <h3>lj98</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/lj98">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/lj98.html#lj98">Full ranked table and performance profile</a>.</p>
   <h3>tip4p10</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/tip4p10">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/tip4p10.html#tip4p10">Full ranked table and performance profile</a>.</p>
   <h3>tip4p15</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/tip4p15">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/tip4p15.html#tip4p15">Full ranked table and performance profile</a>.</p>
   <h3>tip4p20</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="globalopt/tip4p20">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/globalopt/tip4p20.html#tip4p20">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-lowest-eigenvector">Lowest eigenvector</h2>
   <h3>lj38</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="lowest-eigenvector/lj38">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">sella-0.1.1/lj38-loweig</td><td data-field="version">Sella 0.1.1</td><td data-field="commit">0.1.1</td><td data-field="source">campaign, asv</td><td data-field="par2_mean">17.8</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.981155</td><td data-field="wilson_high">1</td><td data-field="median_calls">17</td><td data-field="mean_calls">17.8</td><td data-field="archive">sella: Sella 0.1.1, force calls 17.8</td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">200</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/lowest-eigenvector/lj38.html#lj38">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-tsopt">Transition state optimization</h2>
   <h3>LJ38</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="tsopt/LJ38">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">sella/lj38-ts-default</td><td data-field="version">Sella 2.6.0</td><td data-field="commit">6cfb60cdb100</td><td data-field="source">campaign, asv</td><td data-field="par2_mean">68.74</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.981155</td><td data-field="wilson_high">1</td><td data-field="median_calls">59.5</td><td data-field="mean_calls">68.74</td><td data-field="archive"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">200</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">eOn rows are not yet on the harness counter. Files with no per-start harness record stay off the ranked table.</p>
   <p><a href="generated/suites/tsopt/LJ38.html#LJ38">Full ranked table and performance profile</a>.</p>
   <h3>pt-island</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="tsopt/pt-island">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">sella/pt-island-ts-default</td><td data-field="version">Sella 2.6.0</td><td data-field="commit">6cfb60cdb100</td><td data-field="source">campaign, asv</td><td data-field="par2_mean">74.2857</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.927302</td><td data-field="wilson_high">1</td><td data-field="median_calls">71</td><td data-field="mean_calls">74.2857</td><td data-field="archive"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">49</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">eOn rows are not yet on the harness counter.</p>
   <p><a href="generated/suites/tsopt/pt-island.html#pt-island">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-mlopt">Machine learning optimization</h2>
   <h3>C2H4</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="mlopt/C2H4">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/mlopt/C2H4.html#C2H4">Full ranked table and performance profile</a>.</p>
   <h3>EMT Pt 38 nanoparticle</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="mlopt/emt-pt38">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/mlopt/emt-pt38.html#emt-pt38">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-neb">Nudged elastic band</h2>
   <h3>Pt-heptamer</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="neb/Pt-heptamer">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>Per-image mean</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td data-field="template">rgsaddle/idpp-ci-neb-lbfgs</td><td data-field="version">rgsaddle 0.2.2</td><td data-field="commit">c0db2b8d333c</td><td data-field="source">campaign, asv</td><td data-field="par2_mean">318.308</td><td data-field="success_rate">1</td><td data-field="wilson_low">0.771905</td><td data-field="wilson_high">1</td><td data-field="median_calls">314</td><td data-field="mean_calls">318.308</td><td data-field="mean_per_image">39.7885</td><td data-field="archive"></td><td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td><td data-field="n">13</td></tr>
   </tbody></table></div>
   <p class="bench-gaps">eOn band rows are not yet on the harness counter. Files with no per-start harness record stay off the ranked table.</p>
   <p><a href="generated/suites/neb/Pt-heptamer.html#Pt-heptamer">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-hTST">Harmonic transition state theory</h2>
   <h3>Al</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="hTST/Al">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/hTST/Al.html#Al">Full ranked table and performance profile</a>.</p>
   <h3>Pt</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="hTST/Pt">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/hTST/Pt.html#Pt">Full ranked table and performance profile</a>.</p>
   <h2 id="suite-accelerated-md">Accelerated molecular dynamics</h2>
   <h3>aladatom</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="accelerated-md/aladatom">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/accelerated-md/aladatom.html#aladatom">Full ranked table and performance profile</a>.</p>
   <h3>standard_aladatom</h3>
   <div class="bench-scroll">
   <table data-board-table="headline" data-problem="accelerated-md/standard_aladatom">
   <thead><tr><th>Template</th><th>Version</th><th>Commit</th><th>Source</th><th>PAR-2</th><th>Success</th><th>Wilson low</th><th>Wilson high</th><th>Median calls</th><th>Mean calls</th><th>2014 archive</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr><td colspan="13">No full harness row.</td></tr>
   </tbody></table></div>
   <p><a href="generated/suites/accelerated-md/standard_aladatom.html#standard_aladatom">Full ranked table and performance profile</a>.</p>
   </section>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 2014 problems

   generated/suites/minimization
   generated/suites/saddle-search
   generated/suites/double-ended-saddle-search
   generated/suites/globalopt
   generated/suites/lowest-eigenvector
   generated/suites/tsopt
   generated/suites/mlopt
   generated/suites/neb
   generated/suites/hTST
   generated/suites/accelerated-md
