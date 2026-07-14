# Release Checklist

1. Create a new public GitHub repository.
2. Upload the contents of this folder as the repository root.
3. Confirm that raw images, checkpoints, `.npy` arrays, and cache files are not committed.
4. Run `pytest` in a clean environment.
5. Create a GitHub release, for example `v0.1.0`.
6. Archive the GitHub release with Zenodo.
7. Replace the placeholder URL and DOI in `CITATION.cff`.
8. Replace the manuscript Code Availability statement using `docs/CODE_AVAILABILITY_TEMPLATE.md`.

For a CMPB software-framework paper, the repository and archived DOI should be available before submission.
