First run `scopesim_generate_example.py` to create a test image (or provide your own image in the relevant variable).

Then you can run `photometry_example_vanilla.py`

`util.py` contains common helper functions.

----

Tested with library versions:
```requirements.txt
-e git+git@github.com:krachyon/AnisoCADO.git@d242d0ca02352e780d31e7eb74d99aa9102caf60#egg=anisocado
-e git+git@github.com:krachyon/photutils.git@33a5f272d0be289acfd2a316b2697d7c4b32dd11#egg=photutils
-e git+https://github.com/AstarVienna/ScopeSim_Templates/@9d0596872de3be447005b4542c2196119a2dd573#egg=ScopeSim_Templates
ScopeSim==0.2.0rc1
```
the `photometry_example_my_changes.py` file requires the relevant changes from my branches of `photutils`,
the other one should run with an unmodified version.

