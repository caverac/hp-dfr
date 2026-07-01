/**
 * Root semantic-release configuration.
 *
 * Drives versioning and GitHub Releases from Conventional Commits. There is no
 * package to publish (the workspace packages are internal, not distributions), so
 * `@semantic-release/npm` runs with `npmPublish: false` purely to bump the version
 * in the root package.json -- which the docs site reads to render its version badge.
 * The release commit/tag and the docs deploy that follows it are handled in
 * .github/workflows/release.yml.
 */
export default {
  branches: ['main'],
  tagFormat: 'v${version}',
  plugins: [
    '@semantic-release/commit-analyzer',
    '@semantic-release/release-notes-generator',
    [
      '@semantic-release/changelog',
      {
        changelogFile: 'CHANGELOG.md'
      }
    ],
    [
      '@semantic-release/exec',
      {
        prepareCmd: 'npx prettier --write CHANGELOG.md'
      }
    ],
    [
      '@semantic-release/npm',
      {
        npmPublish: false
      }
    ],
    [
      '@semantic-release/git',
      {
        assets: ['CHANGELOG.md', 'package.json'],
        message: 'chore(release): ${nextRelease.version} [skip ci]'
      }
    ],
    '@semantic-release/github'
  ]
}
