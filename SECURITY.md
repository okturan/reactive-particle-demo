# Security policy

## Supported version

Security fixes target the current `main` branch and the public deployment at
[reactive-particle-demo.okan.workers.dev](https://reactive-particle-demo.okan.workers.dev/).
Historical commits, generated bundles, and prior Worker versions are not
maintained as supported releases.

## Report a vulnerability privately

Use GitHub's [private vulnerability reporting form](https://github.com/okturan/reactive-particle-demo/security/advisories/new).
Private vulnerability reporting is enabled so a report can be reproduced and
fixed without publishing exploit details in an issue.

Include the affected URL, mode, script, or commit; the security impact;
reproducible steps; and any suggested mitigation. Do not attach real camera
footage, biometric data, credentials, or other personal information. Use
synthetic landmarks or a minimal redacted recording whenever possible.

Use a normal GitHub issue for rendering defects, tracking-quality problems,
device compatibility, and feature requests that do not create a security impact.

## Security and data boundary

- Camera frames and MediaPipe inference remain in the browser. The application
  does not upload frames, landmarks, or derived biometric information.
- The public Worker serves static assets only. It has no server-side application
  code, accounts, data bindings, write operations, or repository secrets.
- Hand and face modes download the pinned MediaPipe browser runtime and landmark
  models from documented jsDelivr and Google-hosted origins. A compromised
  dependency, model, or delivery path remains part of the client-side threat model.
- Mouse mode is the no-camera interaction path. Webcam permission is optional and
  should be requested only for the hand and face experiences.
- Public `verify` and synthetic-input query parameters exercise deterministic
  browser behavior. They do not unlock data, credentials, or privileged actions.
- GitHub Actions uses read-only repository permissions, immutable action
  revisions, locked npm dependencies, a production build, and synthetic browser
  verification. Failed checks retain no camera footage or personal data.

Useful reports include unintended camera activation, frame or landmark exfiltration,
cross-site scripting, dependency or workflow compromise, unsafe model loading,
exposed secrets, static-hosting misconfiguration, or a way to turn synthetic
verification controls into access to user data or privileged behavior.

## Disclosure

Please allow time to reproduce and remediate a confirmed issue before public
disclosure. This project does not currently operate a bug-bounty program.
