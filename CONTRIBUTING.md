# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Changes Accepted

Please file issues before doing substantial work; this will ensure that others
don't duplicate the work and that there's a chance to discuss any design issues.

Changes only tweaking style are unlikely to be accepted unless they are applied
consistently across the project. Most of the code style is derived from the
[Google Style Guides](http://google.github.io/styleguide/) for the appropriate
language and is generally not something we accept changes on (as clang-format
and clang-tidy handle that for us). Improvements to code structure and clarity
are welcome but please file issues to track such work first.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests (PRs) for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Peculiarities

We use a GitHub integration to import PRs into our upstream (Google internal)
source code management. Once it is approved internally, each PR will be merged
into the master branch as a single commit by the same tooling. The description
will match the PR title followed by the PR description. Accordingly, please
write these as you would a helpful commit message. Please also keep PRs small
(focused on a single issue) to streamline review and ease later culprit-finding.

Our documentation on
[repository management](https://github.com/google/iree/blob/master/docs/repository_management.md)
has more information on some of the oddities in our repository setup and
workflows.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).
