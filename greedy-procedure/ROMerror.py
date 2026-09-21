import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import subprocess
from joblib import Parallel, delayed
import shutil

# NOTE: run directories are passed around WITH a trailing '/', since getCp
# builds its file paths as '{}postpro/...'.format(runDir).

def count_runs(path, prefix):
    """Number of 'prefix000' run directories under path, ignoring anything else.

    An evaluate/romruns folder can also hold hand-copied directories such as
    point061_Des; counting every subdirectory would inflate the run count and
    send the loop past the last real point.
    """
    pattern = re.compile(r'{}\d{{3}}$'.format(prefix))
    return sum(1 for p in Path(path).iterdir() if p.is_dir() and pattern.match(p.name))

def getCp(runDir, ref_nodes, plot=True):
    ref_pos=ref_nodes

    # Read files, rely on the fact that skinfriction=0.0 for non-surface node
    sfFile = '{}postpro/SkinFriction.xpost'.format(runDir)
    sf = np.loadtxt(sfFile, skiprows = 3)
    idx = np.nonzero(sf)
    sf = sf[idx]

    posFile = '{}deform/Position.xpost'.format(runDir)
    pos = np.loadtxt(posFile, skiprows = 3)
    pos = pos[idx]

    pressFile = '{}postpro/PressureCoefficient.xpost'.format(runDir)
    press = np.loadtxt(pressFile, skiprows = 3)
    press = press[idx]

    ref_pos=ref_pos[idx]

    # Split into top and bottom surfaces
    top_idx = np.nonzero((ref_pos[:,2]>=0)*(ref_pos[:,3]==0.))
    bot_idx = np.nonzero((ref_pos[:,2]<0)*(ref_pos[:,3]==0.))

    pos_top = pos[top_idx]
    pos_bot = pos[bot_idx]

    sf_top = sf[top_idx]
    sf_bot = sf[bot_idx]

    press_top = press[top_idx]
    press_bot = press[bot_idx]

    # Sort by x location
    top_idx = np.argsort(pos_top[:,0])
    bot_idx = np.argsort(-pos_bot[:,0])

    pos_top = pos_top[top_idx]
    pos_bot = pos_bot[bot_idx]

    sf_top = sf_top[top_idx]
    sf_bot = sf_bot[bot_idx]

    press_top = press_top[top_idx]
    press_bot = press_bot[bot_idx]

    #Concatenate arrays
    pos = np.concatenate((pos_top, pos_bot))
    press = np.concatenate((press_top, press_bot))
    sf = np.concatenate((sf_top, sf_bot))

    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(pos_top[:,0], press_top, label='Top Surface')
        plt.plot(pos_bot[:,0], press_bot, label='Bottom Surface')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$C_p$')
        plt.legend()
        plt.savefig('{}postpro/Cp_curve.pdf'.format(runDir))
        plt.close()

    return pos_top[:,0], pos_bot[:,0], press_top, press_bot

def getCp_HDM(HDMdir, ref_nodes):
    pos_top, pos_bot, press_top, press_bot = getCp(HDMdir, ref_nodes, plot=False)
    return pos_top, pos_bot, press_top, press_bot

def sower_sf_cp(dataDir, runDir):
    """Sower SkinFriction and PressureCoefficient for a single run directory.

    dataDir holds the fluidmodel con/msh the run was partitioned with, e.g.
    'combined_Sobol/NonDes_Laplace/GreedyRuns_50/data' for the ROMs or
    'combined_HDMs/GreedyRuns/data' for the truth HDMs.
    """
    print('Sowering for Sf and Cp in: {}'.format(runDir))

    sower='/home/groups/cfarhat/bin/sower'
    con='{}/fluidmodel.con'.format(dataDir)
    msh='{}/fluidmodel.msh'.format(dataDir)

    result_dir='{}results'.format(runDir)
    postpro_dir='{}postpro'.format(runDir)

    for field in ['SkinFriction', 'PressureCoefficient']:
        subprocess.run('{} -fluid -merge -con {} -mesh {} -result {} -output {}'.format(
                           sower, con, msh,
                           '{}/{}.bin'.format(result_dir, field),
                           '{}/{}'.format(postpro_dir, field)),
                       shell=True, check=True, stdout=subprocess.DEVNULL)

def isSowered(runDir, field):
    """True only if the xpost file exists AND has content.

    A sower that dies partway leaves a 0-byte xpost behind. That file exists, so
    an existence-only check would skip re-sowering and np.loadtxt would then
    return an empty array instead of the field.
    """
    path = '{}postpro/{}.xpost'.format(runDir, field)
    return os.path.exists(path) and os.path.getsize(path) > 0

def sowerIfNeeded(dataDir, runDir):
    """Sower Sf/Cp unless both xpost files are already present and non-empty."""
    if not all(isSowered(runDir, field) for field in ['SkinFriction', 'PressureCoefficient']):
        sower_sf_cp(dataDir, runDir)

def readSnappedIndices(snapped_file):
    """Return the set of snapped PROM indices listed in snapped_indices.txt."""
    if not os.path.exists(snapped_file):
        print('Warning: no snapped indices file at {}. '
              'No ROM will be excluded as snapped.'.format(snapped_file))
        return set()

    with open(snapped_file, 'r') as f:
        snapped = set(int(m) for m in re.findall(r'PROM run (\d+)', f.read()))

    print('Read {} snapped PROM indices from {}'.format(len(snapped), snapped_file))
    return snapped

def readParams(runDir):
    """Read a ROM run's parameters.txt (first entry is the parameter count)."""
    values = np.loadtxt('{}parameters.txt'.format(runDir))
    return values[1:]

def getLD(runDir):
    """Lift and drag of a converged run, from the last line of liftdrag.out.

    Column layout is '# TimeIteration Time SubCycles NewtonSteps Lx Ly Lz', with
    the force already rotated into the freestream frame: Lx is drag, Ly is lift.
    Same convention as getLD in runs.py.
    """
    with open('{}postpro/liftdrag.out'.format(runDir), 'r') as f:
        line = f.readlines()[-1].split()

    lift = float(line[5])
    drag = float(line[4])
    return lift, drag

def ldError(romDir, hdmDir):
    """Relative lift and drag error of the ROM against the HDM truth.

    The ROM and the HDM at the same parameter point share the freestream and the
    reference area, so q_inf and S cancel in the ratio and the dimensional forces
    give exactly the same relative error as Cl and Cd would.
    """
    rom_lift, rom_drag = getLD(romDir)
    hdm_lift, hdm_drag = getLD(hdmDir)

    err_cl = abs(rom_lift - hdm_lift) / abs(hdm_lift)
    err_cd = abs(rom_drag - hdm_drag) / abs(hdm_drag)

    return err_cl, err_cd

def cpL2error(romDir, hdmDir, ref_nodes):
    """Relative L2 error of the ROM Cp curve against the HDM truth.

    Returns (err_top, err_bot, err_all). The ROM and the HDM at the same
    parameter point share the mesh and the prescribed deformation, so their
    surface node sets are identical and the x-sorted curves compare node for
    node without interpolation.
    """
    _, _, rom_top, rom_bot = getCp(romDir, ref_nodes, plot=False)
    _, _, hdm_top, hdm_bot = getCp_HDM(hdmDir, ref_nodes)

    if rom_top.shape != hdm_top.shape or rom_bot.shape != hdm_bot.shape:
        raise ValueError('Cp curve length mismatch: ROM ({}, {}) vs HDM ({}, {})'.format(
            rom_top.size, rom_bot.size, hdm_top.size, hdm_bot.size))

    rom_all = np.concatenate((rom_top, rom_bot))
    hdm_all = np.concatenate((hdm_top, hdm_bot))

    err_top = np.linalg.norm(rom_top - hdm_top) / np.linalg.norm(hdm_top)
    err_bot = np.linalg.norm(rom_bot - hdm_bot) / np.linalg.norm(hdm_bot)
    err_all = np.linalg.norm(rom_all - hdm_all) / np.linalg.norm(hdm_all)

    return err_top, err_bot, err_all

def evaluateSinglePoint(folder, testcase, greedyFolder, truthFolder, numHDM, k, ref_nodes, snapped):
    """Compute the Cp error for ROM point k, or report why it was skipped."""
    romProj = os.path.join(folder, testcase)
    romDir = '{}/{}/evaluate/romruns{:03d}/point{:03d}/'.format(romProj, greedyFolder, numHDM, k)
    hdmDir = '{}/InitialHDMruns/HDMrun{:03d}/'.format(truthFolder, k)

    result = {'k': k,
              'params': readParams(romDir),
              'err_top': np.nan,
              'err_bot': np.nan,
              'err_all': np.nan,
              'err_cl': np.nan,
              'err_cd': np.nan,
              'status': 'SNAPPED' if k in snapped else 'OK'}

    if not os.path.exists('{}results/Mach.bin001'.format(romDir)):
        result['status'] = 'ROM FAILED'
        return result

    if not os.path.exists(hdmDir):
        result['status'] = 'HDM PENDING'
        return result

    # each side is sowered with the con/msh it was partitioned with: the ROMs live
    # under the testcase's own greedy folder, the truth HDMs under combined_HDMs/GreedyRuns
    romData = os.path.join(romProj, greedyFolder, 'data')
    hdmData = os.path.join(truthFolder, 'GreedyRuns', 'data')

    for dataDir, runDir in [(romData, romDir), (hdmData, hdmDir)]:
        if not os.path.exists('{}deform/Position.xpost'.format(runDir)):
            result['status'] = 'NO POSITION'
            return result
        sowerIfNeeded(dataDir, runDir)

    try:
        result['err_top'], result['err_bot'], result['err_all'] = cpL2error(romDir, hdmDir, ref_nodes)
        result['err_cl'], result['err_cd'] = ldError(romDir, hdmDir)
    except (ValueError, OSError, IndexError) as e:
        result['status'] = 'READ ERROR'
        print('Warning: point{:03d} skipped: {}'.format(k, e))
        return result

    print('point{:03d}: relative Cp L2 error = {:.4f} % ({})'.format(
        k, 100.0 * result['err_all'], result['status']))
    return result

def computeCpErrors(folder, testcase, greedyFolder, truthFolder, numHDM, numROM, numHDMdone,
                    ref_nodes, snapped, nprocs=1):
    """Evaluate every ROM point that has a finished truth HDM."""
    points = [k for k in range(1, numROM + 1) if k <= numHDMdone]
    pending = [k for k in range(1, numROM + 1) if k > numHDMdone]

    results = Parallel(n_jobs=min(len(points), nprocs))(
        delayed(evaluateSinglePoint)(folder, testcase, greedyFolder, truthFolder, numHDM, k,
                                     ref_nodes, snapped)
        for k in points
    )

    romProj = os.path.join(folder, testcase)
    for k in pending:
        romDir = '{}/{}/evaluate/romruns{:03d}/point{:03d}/'.format(romProj, greedyFolder, numHDM, k)
        results.append({'k': k, 'params': readParams(romDir), 'err_top': np.nan,
                        'err_bot': np.nan, 'err_all': np.nan, 'err_cl': np.nan,
                        'err_cd': np.nan, 'status': 'HDM PENDING'})

    return sorted(results, key=lambda r: r['k'])

def printReport(results, testcase, truthFolder, output_file):
    """Print the per-point table and the summary, echoing everything to output_file.

    The report is written through a single held handle ('w' truncates a previous
    run) rather than reopening the file per line.
    """
    with open(output_file, 'w') as f:

        def emit(line=''):
            print(line)
            f.write(line + '\n')

        emit('\n' + '=' * 78)
        emit('Relative L2 error of the Cp curve: {} ROMs vs {} truth HDMs'.format(testcase, truthFolder))
        emit('=' * 78)
        emit('{:>5}  {:>6} {:>7} {:>6}  {:>10} {:>10} {:>10}  {}'.format(
            'point', 'Mach', 'Beta', 'p3', 'err_top[%]', 'err_bot[%]', 'err_L2[%]', 'flag'))

        for r in results:
            params = r['params']
            errs = '{:>10} {:>10} {:>10}'.format('-', '-', '-') if np.isnan(r['err_all']) else \
                   '{:10.4f} {:10.4f} {:10.4f}'.format(100.0 * r['err_top'], 100.0 * r['err_bot'],
                                                       100.0 * r['err_all'])
            emit('{:>5s}  {:6.3f} {:7.3f} {:6.3f}  {}  {}'.format(
                '{:03d}'.format(r['k']), params[0], params[1], params[2], errs,
                '' if r['status'] == 'OK' else r['status']))

        included = [r for r in results if r['status'] == 'OK' and not np.isnan(r['err_all'])]
        errors = np.array([r['err_all'] for r in included])

        emit('\nSummary (snapped / pending / failed excluded): N = {} of {}'.format(
            len(included), len(results)))
        if included:
            emit('  Cp  mean {:8.4f} %   median {:8.4f} %   min {:8.4f} % (point {:03d})   max {:8.4f} % (point {:03d})'.format(
                100.0 * errors.mean(), 100.0 * np.median(errors),
                100.0 * errors.min(), included[errors.argmin()]['k'],
                100.0 * errors.max(), included[errors.argmax()]['k']))

            # Lift and drag: q_inf and the reference area cancel in the relative error,
            # so these are the Cl and Cd relative errors without any nondimensionalization
            for key, name in [('err_cl', 'Cl'), ('err_cd', 'Cd')]:
                vals = np.array([r[key] for r in included])
                if np.all(np.isnan(vals)):
                    emit('  {}  no liftdrag.out available'.format(name))
                    continue
                emit('  {}  mean {:8.4f} %   median {:8.4f} %   min {:8.4f} % (point {:03d})   max {:8.4f} % (point {:03d})'.format(
                    name, 100.0 * np.nanmean(vals), 100.0 * np.nanmedian(vals),
                    100.0 * np.nanmin(vals), included[int(np.nanargmin(vals))]['k'],
                    100.0 * np.nanmax(vals), included[int(np.nanargmax(vals))]['k']))
        else:
            emit('  no point had both a converged ROM and a finished truth HDM.')

        snapped = [r for r in results if r['status'] == 'SNAPPED']
        emit('\nSnapped ROMs (excluded from the summary): {}'.format(
            ', '.join('{:03d}'.format(r['k']) for r in snapped) if snapped else 'none'))
        for r in snapped:
            if np.isnan(r['err_all']):
                emit('  point {:03d}: truth HDM not available yet'.format(r['k']))
            else:
                emit('  point {:03d}: {:8.4f} %'.format(r['k'], 100.0 * r['err_all']))

        for status in ['ROM FAILED', 'NO POSITION', 'READ ERROR', 'HDM PENDING']:
            flagged = [r['k'] for r in results if r['status'] == status]
            if flagged:
                emit('\n{} ({}): {}'.format(status, len(flagged),
                                            ', '.join('{:03d}'.format(k) for k in flagged)))

    print('\nSaved report to: {}'.format(output_file))


if __name__ == "__main__":
    folder = 'combined_Sobol'
    testcase = 'Des_NoShift' # 'NonDes_NoShift', 'Des_NoShift', 'NonDes_Laplace'
    truthFolder = 'combined_HDMs' # holds the truth HDMs, InitialHDMruns/HDMrun{k:03d} <-> point{k:03d}
    numHDM = 70 # size of the ROM basis, selects romruns{numHDM:03d}
    greedyFolder = 'GreedyRuns_{}'.format(numHDM) # the greedy folder under the testcase, e.g. GreedyRuns_50
    nproc = 4

    numROM = count_runs(os.path.join(folder, testcase, greedyFolder, 'evaluate',
                                     'romruns{:03d}'.format(numHDM)), 'point')
    numHDMdone = count_runs(os.path.join(truthFolder, 'InitialHDMruns'), 'HDMrun')
    print('{} ROM points, {} truth HDMs finished so far'.format(numROM, numHDMdone))

    ref_nodes = np.loadtxt(os.path.join(folder, testcase, 'mesh', 'naca0012_Re1p5_nodes'),
                           dtype=np.float64)
    snapped = readSnappedIndices(os.path.join(folder, testcase, 'snapped_indices_{}.txt'.format(numHDM)))

    results = computeCpErrors(folder, testcase, greedyFolder, truthFolder, numHDM, numROM,
                              numHDMdone, ref_nodes, snapped, nprocs=nproc)

    # one report per testcase, so switching testcase does not overwrite the previous one
    output_file = os.path.join(folder, 'results_{}.ROMerror_{}'.format(numHDM, testcase))
    printReport(results, testcase, truthFolder, output_file)
