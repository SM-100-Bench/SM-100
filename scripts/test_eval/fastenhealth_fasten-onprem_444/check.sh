#!/bin/bash
if grep -Fq "this.elementRef.nativeElement.src = this.imageFallback || 'assets/images/no-image.svg'" frontend/src/app/directives/image-fallback.directive.ts; then
    exit 1
else
    exit 0
fi