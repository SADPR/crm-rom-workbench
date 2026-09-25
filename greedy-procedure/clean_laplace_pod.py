#!/usr/bin/env python3
"""Assemble and reduce only the clean CRM Laplace campaign."""

import argparse

import initial_pod


def configure_settings():
    """Pin the clean campaign roots and the validated ScaLAPACK POD method."""
    settings = initial_pod.configure_settings()
    settings.MasterDir = 'CleanLaplaceRuns/'
    settings.InitHDMPreCompDir = 'CleanLaplacePrecompute/'
    settings.PODMethod = 'ScalapackSVD'
    return settings


def main():
    """Run one safe clean-campaign POD stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('audit', 'assemble', 'pod'))
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'audit':
        initial_pod.audit(settings)
    elif args.mode == 'assemble':
        initial_pod.assemble(settings)
    else:
        initial_pod.build_pod(settings)


if __name__ == '__main__':
    main()
