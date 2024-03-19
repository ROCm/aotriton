#!/bin/bash

dockerimage="$1"
package="$2"

docker run --rm -i "$dockerimage" /bin/sh -c 'cd /opt; tar cj aotriton' > "$package"
