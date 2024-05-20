# Welcome to Quark!

-   [Before you get started](#before-you-get-started)
    -   [Code of Conduct](#code-of-conduct)
    -   [Community Expectations](#community-expectations)
-   [Your First Contribution](#your-first-contribution)
    -   [Find something to work on](#find-something-to-work-on)
        -   [Find a good first topic](#find-a-good-first-topic)
        -   [Work on an Issue](#work-on-an-issue)
        -   [File an Issue](#file-an-issue)
-   [Contributor Workflow](#contributor-workflow)
    -   [Creating Pull Requests](#creating-pull-requests)
    -   [Code Review](#code-review)
    -   [Format of the commit message](#Format-of-the-commit-message)
-   [Communication](#Communication)

# Before you get started

## Code of Conduct (TBA)

## Community Expectations

Quark is a community project driven by its community which strives to promote a
healthy, friendly and productive environment. The goal of the community is to
develop a cloud native secure container runtime. To build a platform at such
scale requires the support of a community with similar aspirations.

# Your First Contribution

We will help you to contribute in different areas like filing issues, developing
features, fixing critical bugs and getting your work reviewed and merged.

## Find something to work on

We are always in need of help, be it fixing documentation, reporting bugs or
writing some code. Look at places where you feel best coding practices aren't
followed, code refactoring is needed or tests are missing. Here is how you get
started.

### Find a good first topic

There are [multiple repositories](https://github.com/QuarkContainer/) within the
Quark organization. Each repository has beginner-friendly issues that provide a
good first issue.
For example, [Quark/Quark](https://github.com/QuarkContainer/Quark) has
[help wanted](https://github.com/QuarkContainer/Quark/labels/help%20wanted)
and [good first issue](https://github.com/QuarkContainer/Quark/labels/good%20first%20issue)
labels for issues that should not need deep knowledge of the system. We can help
new contributors who wish to work on such issues.

Another good way to contribute is to find a documentation improvement, such as a
missing/broken link. Please see below for the workflow.

### Work on an issue

When you are willing to take on an issue, you can assign it to yourself. Just
reply with `/assign` or `/assign @yourself` on an issue, then the robot will
assign the issue to you and your name will present at `Assignees` list.

### File an Issue

While we encourage everyone to contribute code, it is also appreciated when
someone reports an issue. Issues should be filed under the appropriate Quark
sub-repository. A Quark issue should be opened to
[Quark/Quark](https://github.com/QuarkContainer/Quark/issues).

Please follow the prompted submission guidelines while opening an issue.

# Contributor Workflow

Please do not ever hesitate to ask a question or send a pull request.

This is a rough outline of what a contributor's workflow looks like:

- Create a topic branch from where to base the contribution. This is usually main.
- Make commits of logical units.
- Make sure commit messages are in the proper format (see below).
- Push changes in a topic branch to a personal fork of the repository.
- Submit a pull request to [Quark/Quark](https://github.com/QuarkContainer/Quark).
- The PR must receive an approval from at least one maintainers.

## Creating Pull Requests

Pull requests are often called simply "PR". Quark generally follows the standard
[github pull request](https://help.github.com/articles/about-pull-requests/)
process. To submit a proposed change, please develop the code/fix and add new
test cases. After that, run these local verifications before submitting pull
request to predict the pass or fail of continuous integration.

## Code Review

To make it easier for your PR to receive reviews, consider the reviewers will
need you to:

* follow [good rust coding guidelines](https://rustc-dev-guide.rust-lang.org/conventions.html).
* write [good commit messages](https://chris.beams.io/posts/git-commit/).
* break large changes into a logical series of smaller patches which
  individually make easily understandable changes, and in aggregate solve a
  broader issue.
* label PRs with appropriate reviewers: to do this read the messages the bot
  sends you to guide you through the PR process.

## Format of the commit message

We follow a rough convention for commit messages that is designed to answer two
questions: what changed and why. The subject line should feature the what and
the body of the commit should describe the why.

```
scripts: add test codes for metamanager

this add some unit test codes to improve code coverage for metamanager

Fixes #12
```

The format can be described more formally as follows:

```
<subsystem>: <what changed>
<BLANK LINE>
<why this change was made>
<BLANK LINE>
<footer>
```

The first line is the subject and should be no longer than **70 characters**,
the second line is always blank, and other lines should be wrapped at **80
characters.** This allows the message to be easier to read on GitHub as well as
in various git tools.

Note: if your pull request isn't getting enough attention, you can use the reach
out on Slack to get help finding reviewers.

# Communication

- use the [Quark mailinglist](https://lists.sr.ht/~quark/QuarkContainer) by
  simply sending emails to `~quark/QuarkContainer@lists.sr.ht`. To subscribe to
  the list, send an empty email to `~quark/QuarkContainer+subscribe@lists.sr.ht`
  To learn more about using mailinglist, see the [Mailing list etiquette](https://man.sr.ht/lists.sr.ht/etiquette.md).
- there is also a [Slack channel](https://join.slack.com/t/quarksoftgroup/shared_invite/zt-oj7dgqet-6iUXmOnMbqHj4g_XAd_3Mg)
