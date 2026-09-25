#!/usr/bin/env python3
"""Prepare or run the nominal baseline in the clean CRM campaign."""

import argparse

import baseline_steady


def configure_settings():
    """Keep the clean baseline independent from the exploratory baseline."""
    settings = baseline_steady.configure_settings()
    settings.MasterDir = 'CleanLaplaceBaseline/'
    return settings


def main():
    """Run one explicit baseline stage with the supplied nominal point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--point', nargs=5, type=float,
                        default=baseline_steady.DEFAULT_POINT,
                        metavar=('MACH', 'AOA', 'CAMBER_LOC', 'CAMBER', 'THICKNESS'))
    args = parser.parse_args()

    settings = configure_settings()
    if args.mode == 'prepare':
        baseline_steady.prepare(settings, args.point)
    else:
        baseline_steady.run(settings)


if __name__ == '__main__':
    main()
