=============
Current codes
=============


The table is the registry pin and the ranked harness runs for that code.
A code with no full run on a problem is listed with the reason.
The sections each client reads, and the commands that rerun a score, are in the README and on the Run a score page.
A timestep or a band method in the wrong section is ignored.
The Sella 0.1.1 rows stay on the 2014 tables and are labelled legacy.
A new score is entered by Rohit Goswami.
The number is written only after the run finishes under the problem rule.

.. raw:: html

   <section class="bench" id="current-codes">
   <h2>Current codes</h2>
   <p>The benchmark contributor is Rohit Goswami. A ranked row is a full harness run. The commit on that row is the commit that produced the score.</p>
   <h3>eOn</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>minimization and saddle search</td>
   <td>3.5.0</td>
   <td>4d8b34baf1ba</td>
   <td><a href="https://github.com/TheochemUI/eon">https://github.com/TheochemUI/eon</a></td>
   </tr>
   <tr>
   <td>minimization and saddle search</td>
   <td>main</td>
   <td>1d71c9445018</td>
   <td><a href="https://github.com/TheochemUI/eon">https://github.com/TheochemUI/eon</a></td>
   </tr>
   <tr>
   <td>minimization and saddle search</td>
   <td>develop</td>
   <td>7095b23ac17d</td>
   <td><a href="https://github.com/TheochemUI/eon">https://github.com/TheochemUI/eon</a></td>
   </tr>
   </tbody></table></div>
   <div class="bench-scroll"><table data-board-table="code-best">
   <caption>Best full harness run on each problem.</caption>
   <thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th><th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr data-problem="double-ended-saddle-search/lj38" data-template="eon-client/de-lj38-lanczos-improved">
   <td>LJ38 double-ended saddle</td>
   <td data-field="template">eon-client/de-lj38-lanczos-improved</td>
   <td data-field="version">eOn 3.5.0</td>
   <td data-field="commit">4d8b34baf1ba</td>
   <td data-field="par2_mean">259.46</td>
   <td data-field="mean_calls">259.46</td>
   <td data-field="median_calls">212</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">50</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/lj38_2ts_separation" data-template="eon-client/de-2ts-lanczos-improved">
   <td>lj38_2ts_separation</td>
   <td data-field="template">eon-client/de-2ts-lanczos-improved</td>
   <td data-field="version">eOn 3.5.0</td>
   <td data-field="commit">4d8b34baf1ba</td>
   <td data-field="par2_mean">520.333</td>
   <td data-field="mean_calls">520.333</td>
   <td data-field="median_calls">422</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">51</td>
   </tr>
   <tr data-problem="lowest-eigenvector/lj38" data-template="eon-client/low-lj38-lanczos">
   <td>lj38</td>
   <td data-field="template">eon-client/low-lj38-lanczos</td>
   <td data-field="version">eOn 3.5.0</td>
   <td data-field="commit">4d8b34baf1ba</td>
   <td data-field="par2_mean">26.385</td>
   <td data-field="mean_calls">26.385</td>
   <td data-field="median_calls">25</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">200</td>
   </tr>
   <tr data-problem="minimization/lj38" data-template="eon/lj38-lbfgs-m100">
   <td>Lennard-Jones 38 clusters</td>
   <td data-field="template">eon/lj38-lbfgs-m100</td>
   <td data-field="version">eOn 3.6.0</td>
   <td data-field="commit">1d71c9445018</td>
   <td data-field="par2_mean">180.137</td>
   <td data-field="mean_calls">180.137</td>
   <td data-field="median_calls">171</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/morse-bulk" data-template="eon/morse-lbfgs-m100">
   <td>Morse bulk</td>
   <td data-field="template">eon/morse-lbfgs-m100</td>
   <td data-field="version">eOn 3.6.0</td>
   <td data-field="commit">1d71c9445018</td>
   <td data-field="par2_mean">54.18</td>
   <td data-field="mean_calls">54.18</td>
   <td data-field="median_calls">53</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">100</td>
   </tr>
   <tr data-problem="tsopt/LJ38" data-template="eon-client/ts-lj38-lanczos-improved">
   <td>LJ38</td>
   <td data-field="template">eon-client/ts-lj38-lanczos-improved</td>
   <td data-field="version">eOn 3.5.0</td>
   <td data-field="commit">4d8b34baf1ba</td>
   <td data-field="par2_mean">280.33</td>
   <td data-field="mean_calls">280.33</td>
   <td data-field="median_calls">214</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">200</td>
   </tr>
   <tr data-problem="tsopt/pt-island" data-template="eon-client/ts-pt-lanczos-improved">
   <td>pt-island</td>
   <td data-field="template">eon-client/ts-pt-lanczos-improved</td>
   <td data-field="version">eOn 3.5.0</td>
   <td data-field="commit">4d8b34baf1ba</td>
   <td data-field="par2_mean">108.796</td>
   <td data-field="mean_calls">108.796</td>
   <td data-field="median_calls">107</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">49</td>
   </tr>
   </tbody></table></div>
   <p>No full ranked run</p><ul>
   <li>aladatom: partial.</li>
   <li>LJ38 double-ended saddle: not yet on the harness counter, partial.</li>
   <li>lj38_2ts_separation: partial.</li>
   <li>pt-island: not yet on the harness counter, partial.</li>
   <li>Al: partial.</li>
   <li>Pt: partial.</li>
   <li>lj38: partial.</li>
   <li>Pt-heptamer: not yet on the harness counter, partial.</li>
   <li>Pt heptamer island: not yet on the harness counter, partial.</li>
   <li>LJ38: not yet on the harness counter, partial.</li>
   <li>pt-island: not yet on the harness counter, partial.</li>
   <li>eon-ci-neb: no per-start record.</li>
   <li>eon-ci-swdneb: no per-start record.</li>
   <li>eon-cg: no per-start record.</li>
   <li>eon-fire: no per-start record.</li>
   <li>eon-lbfgs: no per-start record.</li>
   <li>eon-qm: no per-start record.</li>
   <li>eon-client/cg: no per-start record.</li>
   <li>eon-client/fire: no per-start record.</li>
   <li>eon-client/lbfgs-m100: no per-start record.</li>
   <li>eon-client/lbfgs-m20: no per-start record.</li>
   <li>eon-client/morse-cg: no per-start record.</li>
   <li>eon-client/morse-fire: no per-start record.</li>
   <li>eon-client/morse-lbfgs: no per-start record.</li>
   <li>eon-client/morse-sd: no per-start record.</li>
   <li>eon-client/qm: no per-start record.</li>
   <li>eon-client/sd-twopoint: no per-start record.</li>
   <li>eon-client/sd: no per-start record.</li>
   </ul>
   <h3>Eon svn</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>minimization and saddle search</td>
   <td>r2549</td>
   <td>r2549</td>
   <td><a href="https://theory.cm.utexas.edu/svn/eon">https://theory.cm.utexas.edu/svn/eon</a></td>
   </tr>
   </tbody></table></div>
   <div class="bench-scroll"><table data-board-table="code-best">
   <caption>Best full harness run on each problem.</caption>
   <thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th><th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr data-problem="double-ended-saddle-search/lj38" data-template="eon-svn/de-lj38-lanczos-improved">
   <td>LJ38 double-ended saddle</td>
   <td data-field="template">eon-svn/de-lj38-lanczos-improved</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">235.2</td>
   <td data-field="mean_calls">235.2</td>
   <td data-field="median_calls">197</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">50</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/lj38_2ts_separation" data-template="eon-svn/de-2ts-lanczos-improved">
   <td>lj38_2ts_separation</td>
   <td data-field="template">eon-svn/de-2ts-lanczos-improved</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">502.784</td>
   <td data-field="mean_calls">502.784</td>
   <td data-field="median_calls">418</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">51</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/pt-island" data-template="eon-svn/de-pt-lanczos-improved">
   <td>pt-island</td>
   <td data-field="template">eon-svn/de-pt-lanczos-improved</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">180.644</td>
   <td data-field="mean_calls">180.644</td>
   <td data-field="median_calls">161</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">59</td>
   </tr>
   <tr data-problem="lowest-eigenvector/lj38" data-template="eon-svn/low-lj38-lanczos-ljcut">
   <td>lj38</td>
   <td data-field="template">eon-svn/low-lj38-lanczos-ljcut</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">26.385</td>
   <td data-field="mean_calls">26.385</td>
   <td data-field="median_calls">25</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">200</td>
   </tr>
   <tr data-problem="minimization/lj38-noise" data-template="eon-svn/lj38-noise-0">
   <td>Lennard-Jones 38 clusters with Gaussian force noise</td>
   <td data-field="template">eon-svn/lj38-noise-0</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">300.052</td>
   <td data-field="mean_calls">200.9</td>
   <td data-field="median_calls">189</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 5</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="neb/Pt-heptamer" data-template="eon-svn/neb-pt-ci">
   <td>Pt-heptamer</td>
   <td data-field="template">eon-svn/neb-pt-ci</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">336.769</td>
   <td data-field="mean_calls">336.769</td>
   <td data-field="median_calls">338</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">13</td>
   </tr>
   <tr data-problem="tsopt/LJ38" data-template="eon-svn/ts-lj38-lanczos-improved">
   <td>LJ38</td>
   <td data-field="template">eon-svn/ts-lj38-lanczos-improved</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">270.85</td>
   <td data-field="mean_calls">270.85</td>
   <td data-field="median_calls">180</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">200</td>
   </tr>
   <tr data-problem="tsopt/pt-island" data-template="eon-svn/ts-pt-lanczos-improved">
   <td>pt-island</td>
   <td data-field="template">eon-svn/ts-pt-lanczos-improved</td>
   <td data-field="version">Eon svn r2549</td>
   <td data-field="commit">r2549</td>
   <td data-field="par2_mean">106.388</td>
   <td data-field="mean_calls">106.388</td>
   <td data-field="median_calls">100</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">49</td>
   </tr>
   </tbody></table></div>
   <p>No full ranked run</p><ul>
   <li>aladatom: partial.</li>
   <li>lj38_2ts_separation: partial.</li>
   <li>Lennard-Jones 38 global minimum: partial.</li>
   <li>Al: partial.</li>
   <li>Pt: partial.</li>
   <li>lj38: partial.</li>
   <li>Pt heptamer island: partial.</li>
   <li>LJ38: partial.</li>
   <li>pt-island: partial.</li>
   <li>eon-svn/cg: no per-start record.</li>
   <li>eon-svn/fire: no per-start record.</li>
   <li>eon-svn/lbfgs-m100: no per-start record.</li>
   <li>eon-svn/lbfgs-m20: no per-start record.</li>
   <li>eon-svn/morse-cg: no per-start record.</li>
   <li>eon-svn/morse-fire: no per-start record.</li>
   <li>eon-svn/morse-lbfgs: no per-start record.</li>
   <li>eon-svn/morse-sd: no per-start record.</li>
   <li>eon-svn/qm: no per-start record.</li>
   <li>eon-svn/sd-twopoint: no per-start record.</li>
   <li>eon-svn/sd: no per-start record.</li>
   </ul>
   <h3>rgmin</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>minimization</td>
   <td>fix/default-cap-per-method</td>
   <td>f15726f1c798</td>
   <td><a href="https://github.com/OmniPotentRPC/rgmin">https://github.com/OmniPotentRPC/rgmin</a></td>
   </tr>
   </tbody></table></div>
   <div class="bench-scroll"><table data-board-table="code-best">
   <caption>Best full harness run on each problem.</caption>
   <thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th><th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr data-problem="minimization/lj38" data-template="rgmin/lj38-dogleg">
   <td>Lennard-Jones 38 clusters</td>
   <td data-field="template">rgmin/lj38-dogleg</td>
   <td data-field="version">rgmin 0.4.1</td>
   <td data-field="commit">9a2e7ff11fa6</td>
   <td data-field="par2_mean">81.881</td>
   <td data-field="mean_calls">81.881</td>
   <td data-field="median_calls">79</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/lj38-noise" data-template="noise/rgmin-dogleg-s0">
   <td>Lennard-Jones 38 clusters with Gaussian force noise</td>
   <td data-field="template">noise/rgmin-dogleg-s0</td>
   <td data-field="version">rgmin 0.4.1</td>
   <td data-field="commit">9a2e7ff11fa6</td>
   <td data-field="par2_mean">81.881</td>
   <td data-field="mean_calls">81.881</td>
   <td data-field="median_calls">79</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/morse-bulk" data-template="rgmin/morse-newton">
   <td>Morse bulk</td>
   <td data-field="template">rgmin/morse-newton</td>
   <td data-field="version">rgmin 0.4.1</td>
   <td data-field="commit">9a2e7ff11fa6</td>
   <td data-field="par2_mean">8.35</td>
   <td data-field="mean_calls">8.35</td>
   <td data-field="median_calls">9</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">100</td>
   </tr>
   </tbody></table></div>
   <h3>rgsaddle</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>saddle search</td>
   <td>main</td>
   <td>c0db2b8d333c</td>
   <td><a href="https://github.com/OmniPotentRPC/rgsaddle">https://github.com/OmniPotentRPC/rgsaddle</a></td>
   </tr>
   </tbody></table></div>
   <div class="bench-scroll"><table data-board-table="code-best">
   <caption>Best full harness run on each problem.</caption>
   <thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th><th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr data-problem="double-ended-saddle-search/lj38" data-template="rgsaddle/ew-ci-neb-lbfgs">
   <td>LJ38 double-ended saddle</td>
   <td data-field="template">rgsaddle/ew-ci-neb-lbfgs</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">853.56</td>
   <td data-field="mean_calls">673.94</td>
   <td data-field="median_calls">529</td>
   <td data-field="failures">crash 0, divergence 1, cap 2, wrong_stationary 0</td>
   <td data-field="n">50</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/lj38_2ts_separation" data-template="rgsaddle/ew-ci-neb-lbfgs">
   <td>lj38_2ts_separation</td>
   <td data-field="template">rgsaddle/ew-ci-neb-lbfgs</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">8024.43</td>
   <td data-field="mean_calls">4639.8</td>
   <td data-field="median_calls">2414</td>
   <td data-field="failures">crash 0, divergence 0, cap 16, wrong_stationary 1</td>
   <td data-field="n">51</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/pt-island" data-template="rgsaddle/ew-ci-neb-lbfgs">
   <td>pt-island</td>
   <td data-field="template">rgsaddle/ew-ci-neb-lbfgs</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">422.136</td>
   <td data-field="mean_calls">224.017</td>
   <td data-field="median_calls">226</td>
   <td data-field="failures">crash 0, divergence 2, cap 0, wrong_stationary 0</td>
   <td data-field="n">59</td>
   </tr>
   <tr data-problem="lowest-eigenvector/lj38" data-template="rgsaddle/mode-lanczos">
   <td>lj38</td>
   <td data-field="template">rgsaddle/mode-lanczos</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">26.12</td>
   <td data-field="mean_calls">26.12</td>
   <td data-field="median_calls">26</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">200</td>
   </tr>
   <tr data-problem="neb/Pt-heptamer" data-template="rgsaddle/idpp-ci-neb-lbfgs">
   <td>Pt-heptamer</td>
   <td data-field="template">rgsaddle/idpp-ci-neb-lbfgs</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">318.308</td>
   <td data-field="mean_calls">318.308</td>
   <td data-field="median_calls">314</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">13</td>
   </tr>
   <tr data-problem="saddle-search/pt-island" data-template="rgsaddle/saddle-dimer-rotskip-tuned">
   <td>Pt heptamer island</td>
   <td data-field="template">rgsaddle/saddle-dimer-rotskip-tuned</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">10127.1</td>
   <td data-field="mean_calls">619.017</td>
   <td data-field="median_calls">550</td>
   <td data-field="failures">crash 0, divergence 17717, cap 2246, wrong_stationary 35</td>
   <td data-field="n">20000</td>
   </tr>
   <tr data-problem="tsopt/LJ38" data-template="rgsaddle/prfo-tuned">
   <td>LJ38</td>
   <td data-field="template">rgsaddle/prfo-tuned</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">192.6</td>
   <td data-field="mean_calls">192.6</td>
   <td data-field="median_calls">173</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">200</td>
   </tr>
   <tr data-problem="tsopt/pt-island" data-template="rgsaddle/pt-dimer-rotskip">
   <td>pt-island</td>
   <td data-field="template">rgsaddle/pt-dimer-rotskip</td>
   <td data-field="version">rgsaddle 0.2.2</td>
   <td data-field="commit">c0db2b8d333c</td>
   <td data-field="par2_mean">223.898</td>
   <td data-field="mean_calls">223.898</td>
   <td data-field="median_calls">145</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">49</td>
   </tr>
   </tbody></table></div>
   <h3>ASE</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>minimization and saddle search</td>
   <td>3.29.0</td>
   <td>f27c0005ae6a</td>
   <td><a href="https://gitlab.com/ase/ase">https://gitlab.com/ase/ase</a></td>
   </tr>
   </tbody></table></div>
   <div class="bench-scroll"><table data-board-table="code-best">
   <caption>Best full harness run on each problem.</caption>
   <thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th><th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr data-problem="double-ended-saddle-search/lj38" data-template="ase/de-lj38-tuned">
   <td>LJ38 double-ended saddle</td>
   <td data-field="template">ase/de-lj38-tuned</td>
   <td data-field="version">ASE 3.29.0</td>
   <td data-field="commit">f27c0005ae6a</td>
   <td data-field="par2_mean">3463.64</td>
   <td data-field="mean_calls">2014.22</td>
   <td data-field="median_calls">2501</td>
   <td data-field="failures">crash 0, divergence 0, cap 29, wrong_stationary 0</td>
   <td data-field="n">50</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/lj38_2ts_separation" data-template="ase/de-lj38-2ts-default">
   <td>lj38_2ts_separation</td>
   <td data-field="template">ase/de-lj38-2ts-default</td>
   <td data-field="version">ASE 3.29.0</td>
   <td data-field="commit">f27c0005ae6a</td>
   <td data-field="par2_mean">20000</td>
   <td data-field="mean_calls">2500</td>
   <td data-field="median_calls">2500</td>
   <td data-field="failures">crash 0, divergence 51, cap 0, wrong_stationary 0</td>
   <td data-field="n">51</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/pt-island" data-template="ase/de-pt-island-tuned">
   <td>pt-island</td>
   <td data-field="template">ase/de-pt-island-tuned</td>
   <td data-field="version">ASE 3.29.0</td>
   <td data-field="commit">f27c0005ae6a</td>
   <td data-field="par2_mean">847.678</td>
   <td data-field="mean_calls">593.864</td>
   <td data-field="median_calls">374</td>
   <td data-field="failures">crash 0, divergence 0, cap 5, wrong_stationary 0</td>
   <td data-field="n">59</td>
   </tr>
   <tr data-problem="minimization/lj38" data-template="ase/lj38-lbfgs-tuned">
   <td>Lennard-Jones 38 clusters</td>
   <td data-field="template">ase/lj38-lbfgs-tuned</td>
   <td data-field="version">ASE 3.29.0</td>
   <td data-field="commit">f27c0005ae6a</td>
   <td data-field="par2_mean">395.645</td>
   <td data-field="mean_calls">375.645</td>
   <td data-field="median_calls">319</td>
   <td data-field="failures">crash 0, divergence 0, cap 2, wrong_stationary 0</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/lj38-noise" data-template="noise/ase-lbfgs-s0">
   <td>Lennard-Jones 38 clusters with Gaussian force noise</td>
   <td data-field="template">noise/ase-lbfgs-s0</td>
   <td data-field="version">ASE 3.29.0</td>
   <td data-field="commit">f27c0005ae6a</td>
   <td data-field="par2_mean">396.55</td>
   <td data-field="mean_calls">326.761</td>
   <td data-field="median_calls">256.5</td>
   <td data-field="failures">crash 0, divergence 0, cap 5, wrong_stationary 1</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/morse-bulk" data-template="ase/morse-bfgslinesearch-default">
   <td>Morse bulk</td>
   <td data-field="template">ase/morse-bfgslinesearch-default</td>
   <td data-field="version">ASE 3.29.0</td>
   <td data-field="commit">f27c0005ae6a</td>
   <td data-field="par2_mean">28.91</td>
   <td data-field="mean_calls">28.91</td>
   <td data-field="median_calls">28</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">100</td>
   </tr>
   <tr data-problem="neb/Pt-heptamer" data-template="ase/neb-pt-heptamer-tuned">
   <td>Pt-heptamer</td>
   <td data-field="template">ase/neb-pt-heptamer-tuned</td>
   <td data-field="version">ASE 3.29.0</td>
   <td data-field="commit">f27c0005ae6a</td>
   <td data-field="par2_mean">442.615</td>
   <td data-field="mean_calls">442.615</td>
   <td data-field="median_calls">450</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">13</td>
   </tr>
   </tbody></table></div>
   <p>No full ranked run</p><ul>
   <li>LJ38 double-ended saddle: partial.</li>
   <li>lj38_2ts_separation: partial.</li>
   <li>pt-island: partial.</li>
   <li>Lennard-Jones 38 clusters: partial.</li>
   <li>Morse bulk: partial.</li>
   <li>Pt-heptamer: partial.</li>
   <li>Pt heptamer island: partial.</li>
   </ul>
   <h3>Sella</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>minimization and saddle search</td>
   <td>2.6.0</td>
   <td>6cfb60cdb100</td>
   <td><a href="https://github.com/zadorlab/sella">https://github.com/zadorlab/sella</a></td>
   </tr>
   </tbody></table></div>
   <div class="bench-scroll"><table data-board-table="code-best">
   <caption>Best full harness run on each problem.</caption>
   <thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th><th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr data-problem="double-ended-saddle-search/lj38" data-template="sella/de-lj38-tuned">
   <td>LJ38 double-ended saddle</td>
   <td data-field="template">sella/de-lj38-tuned</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">1753.96</td>
   <td data-field="mean_calls">1753.96</td>
   <td data-field="median_calls">2051.5</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">50</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/lj38_2ts_separation" data-template="sella/de-lj38-2ts-tuned">
   <td>lj38_2ts_separation</td>
   <td data-field="template">sella/de-lj38-2ts-tuned</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">30105.9</td>
   <td data-field="mean_calls">30105.9</td>
   <td data-field="median_calls">32141</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">51</td>
   </tr>
   <tr data-problem="double-ended-saddle-search/pt-island" data-template="sella/de-pt-island-tuned">
   <td>pt-island</td>
   <td data-field="template">sella/de-pt-island-tuned</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">615.153</td>
   <td data-field="mean_calls">615.153</td>
   <td data-field="median_calls">374</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">59</td>
   </tr>
   <tr data-problem="minimization/lj38" data-template="sella/lj38-min-tuned">
   <td>Lennard-Jones 38 clusters</td>
   <td data-field="template">sella/lj38-min-tuned</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">253.921</td>
   <td data-field="mean_calls">253.921</td>
   <td data-field="median_calls">248</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/lj38-noise" data-template="noise/sella-min-s0">
   <td>Lennard-Jones 38 clusters with Gaussian force noise</td>
   <td data-field="template">noise/sella-min-s0</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">729.433</td>
   <td data-field="mean_calls">710.211</td>
   <td data-field="median_calls">691.5</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 1</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/morse-bulk" data-template="sella/morse-min-default">
   <td>Morse bulk</td>
   <td data-field="template">sella/morse-min-default</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">28.89</td>
   <td data-field="mean_calls">28.89</td>
   <td data-field="median_calls">28</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">100</td>
   </tr>
   <tr data-problem="saddle-search/pt-island" data-template="sella/pt-island-search-default">
   <td>Pt heptamer island</td>
   <td data-field="template">sella/pt-island-search-default</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">5201.62</td>
   <td data-field="mean_calls">247.812</td>
   <td data-field="median_calls">225</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 4</td>
   <td data-field="n">16</td>
   </tr>
   <tr data-problem="tsopt/LJ38" data-template="sella/lj38-ts-default">
   <td>LJ38</td>
   <td data-field="template">sella/lj38-ts-default</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">68.74</td>
   <td data-field="mean_calls">68.74</td>
   <td data-field="median_calls">59.5</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">200</td>
   </tr>
   <tr data-problem="tsopt/pt-island" data-template="sella/pt-island-ts-default">
   <td>pt-island</td>
   <td data-field="template">sella/pt-island-ts-default</td>
   <td data-field="version">Sella 2.6.0</td>
   <td data-field="commit">6cfb60cdb100</td>
   <td data-field="par2_mean">74.2857</td>
   <td data-field="mean_calls">74.2857</td>
   <td data-field="median_calls">71</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">49</td>
   </tr>
   </tbody></table></div>
   <p>No full ranked run</p><ul>
   <li>LJ38 double-ended saddle: partial.</li>
   <li>lj38_2ts_separation: partial.</li>
   <li>pt-island: partial.</li>
   <li>Lennard-Jones 38 clusters: partial.</li>
   <li>Morse bulk: partial.</li>
   <li>Pt heptamer island: partial.</li>
   <li>LJ38: partial.</li>
   <li>pt-island: partial.</li>
   <li>sella-lj38: no per-start record.</li>
   </ul>
   <h3>pARTn</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>saddle search</td>
   <td>DEVEL</td>
   <td>9b753517c6c0</td>
   <td><a href="https://gitlab.com/mammasmias/artn-plugin">https://gitlab.com/mammasmias/artn-plugin</a></td>
   </tr>
   </tbody></table></div>
   <p>No full ranked run.</p>
   <h3>OPTIM</h3>
   <div class="bench-scroll"><table>
   <caption>Registry pins.</caption>
   <thead><tr><th>Role</th><th>Ref</th><th>Commit</th><th>Source</th></tr></thead><tbody>
   <tr>
   <td>minimization</td>
   <td>svn-export</td>
   <td>02aaf45245af</td>
   <td><a href="http://www-wales.ch.cam.ac.uk/svn.tar.bz2">http://www-wales.ch.cam.ac.uk/svn.tar.bz2</a></td>
   </tr>
   </tbody></table></div>
   <div class="bench-scroll"><table data-board-table="code-best">
   <caption>Best full harness run on each problem.</caption>
   <thead><tr><th>Problem</th><th>Template</th><th>Version</th><th>Commit</th><th>PAR-2</th><th>Mean calls</th><th>Median calls</th><th>Failures</th><th>n</th></tr></thead><tbody>
   <tr data-problem="minimization/lj38" data-template="optim/lj38-lbfgs-tuned">
   <td>Lennard-Jones 38 clusters</td>
   <td data-field="template">optim/lj38-lbfgs-tuned</td>
   <td data-field="version">optim 02aaf45245af</td>
   <td data-field="commit">02aaf45245af</td>
   <td data-field="par2_mean">175.537</td>
   <td data-field="mean_calls">175.537</td>
   <td data-field="median_calls">164</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/lj38-noise" data-template="optim/noise-lbfgs-s0">
   <td>Lennard-Jones 38 clusters with Gaussian force noise</td>
   <td data-field="template">optim/noise-lbfgs-s0</td>
   <td data-field="version">optim 02aaf45245af</td>
   <td data-field="commit">02aaf45245af</td>
   <td data-field="par2_mean">290.805</td>
   <td data-field="mean_calls">231.531</td>
   <td data-field="median_calls">218.5</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 3</td>
   <td data-field="n">1000</td>
   </tr>
   <tr data-problem="minimization/morse-bulk" data-template="optim/morse-lbfgs-tuned">
   <td>Morse bulk</td>
   <td data-field="template">optim/morse-lbfgs-tuned</td>
   <td data-field="version">optim 02aaf45245af</td>
   <td data-field="commit">02aaf45245af</td>
   <td data-field="par2_mean">26.66</td>
   <td data-field="mean_calls">26.66</td>
   <td data-field="median_calls">27</td>
   <td data-field="failures">crash 0, divergence 0, cap 0, wrong_stationary 0</td>
   <td data-field="n">100</td>
   </tr>
   </tbody></table></div>
   </section>
