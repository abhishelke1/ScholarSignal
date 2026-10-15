# Dataset provenance

Source: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).

Credit: Cortez, Paulo (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Downloaded from `https://archive.ics.uci.edu/static/public/320/student+performance.zip` on 2026-10-15. The outer archive contains `student.zip`; the unpacked files are supplied here unchanged. `student.txt` is the source data dictionary. The raw archives are retained for provenance.

The app uses only `student-mat.csv` (395 mathematics records), not the Portuguese records. These subjects have overlapping students and must not simply be concatenated for random-split evaluation.

Transformations inside the app: selected predictor subset; exact full-row duplicate removal; final-grade bands derived from G3; fixed categorical bins; dedicated unknown-feature categories. Original CSV files are not modified.

The source text describes the failures field inconsistently; the actual bundled mathematics data has values 0, 1, 2, and 3. This project's input contract follows those observed values.
