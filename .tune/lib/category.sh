#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The task category on the command line of the shared DEF scripts:
# `--category <level0>[,<level1>]`, e.g. tune,kernel, or tune,op. It
# has the same syntax as python/pgdef/pq/category.py.
#
# Protocol: level 0 IS the application's name. Everything in category
# `tune,...` belongs to aotriton.tune, so these scripts take no separate --apps.
#
# End-to-end forwarding: a script that passes the category on to another
# script, over ssh, or into a container, forwards the original string as it
# is ("$CATEGORY"). It never rebuilds the string from the parsed levels, which
# would silently drop any level it did not parse. Each script parses it only
# for the levels it uses itself. The endpoint that gives a level its meaning
# validates it; for example, the worker node's readers use level 1 to claim
# tasks. A user-facing entry script may also validate early, so that a typo
# fails before any ssh or container start; that check is for speed only.

# parse_category <category>
#
# Pure syntax, with no policy and no manifest lookups. Sets CAT_LEVEL0 (the
# application) and CAT_LEVEL1 (empty if absent), each a single comma-separated
# field; any deeper level is left to whoever defines it. An empty category
# leaves both empty. Always returns 0.
parse_category() {
    local category="$1" rest
    CAT_LEVEL0="${category%%,*}"
    CAT_LEVEL1=""
    if [ "$category" != "${category#*,}" ]; then
        rest="${category#*,}"
        CAT_LEVEL1="${rest%%,*}"
    fi
    return 0
}

# validate_category <category>
#
# Policy, from the application's manifest. Source .tune/lib/manifest.sh first.
# Returns 1, after printing an error, if:
#   - the category is empty;
#   - its application is not registered (it has no manifest);
#   - the manifest lists level-1 values and the category gives none, or gives
#     one not in the list (tune needs tune,kernel or tune,op);
#   - the manifest lists none and the category gives one (an application with no levels takes none);
#   - the category has more than two levels (none is defined beyond level 1).
# Call it where a script gives the levels meaning, or as the early check of a
# user-facing entry script. Pure forwarders do not call it.
validate_category() {
    local category="$1" values CAT_LEVEL0 CAT_LEVEL1
    if [ -z "$category" ]; then
        echo "Error: --category <app>[,<level1>] is required (e.g. tune,kernel or tune,op)" >&2
        return 1
    fi
    parse_category "$category"
    if [ -n "$CAT_LEVEL1" ] && [ "${category#*,}" != "$CAT_LEVEL1" ]; then
        echo "Error: invalid --category '$category': only two levels are defined" >&2
        return 1
    fi
    require_app "$CAT_LEVEL0" || return 1
    # (captured first: `grep -q` in a pipeline would SIGPIPE its producer under pipefail)
    values="$(app_manifest "$CAT_LEVEL0" category.level1_values 2>/dev/null || true)"
    if [ -n "$values" ]; then
        if [ -z "$CAT_LEVEL1" ] || ! grep -qx -- "$CAT_LEVEL1" <<<"$values"; then
            echo "Error: invalid --category '$category': application '$CAT_LEVEL0' needs one of:" \
                 $(sed "s/^/$CAT_LEVEL0,/" <<<"$values") >&2
            return 1
        fi
    elif [ -n "$CAT_LEVEL1" ]; then
        echo "Error: invalid --category '$category': application '$CAT_LEVEL0' has no second level" >&2
        return 1
    fi
}
