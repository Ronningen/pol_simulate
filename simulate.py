import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from jet_image import JetImage, TwinJetImage
from vlbi_utils import find_image_std, find_bbox, pol_mask, correct_ppol_bias
sys.path.insert(0, 've/vlbi_errors')
from uv_data import UVData
from spydiff import clean_difmap
from from_fits import create_clean_image_from_fits_file
from image import plot as iplot
import matplotlib.pyplot as plt


def plot_function(contours=None, colors=None, vectors=None, vectors_values=None,
                  cmap='gist_rainbow', abs_levels=None, rel_levels=None, min_abs_level=None,
                  min_rel_level=None, k=2, vinc=2, contours_mask=None, colors_mask=None,
                  vectors_mask=None, color_clim=None, outfile=None, outdir=None, close=False,
                  colorbar_label=None, show=True, contour_color='k', vector_color="k", plot_colorbar=True,
                  max_vector_value_length=5., mas_in_pixel=None, vector_enlarge_factor=1.0,
                  label_size=14, figsize=(20, 5), fig=None, contour_linewidth=1.0, quiver_linewidth=1.0, plot_title=None):
    """
    :param contours: (optional)
        Numpy 2D array (possibly masked) that should be plotted using contours.
    :param colors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using colors.
    :param vectors: (optional)
        Numpy 2D array (possibly masked) that should be plotted using vectors.
    :param vectors_values: (optional)
        Numpy 2D array (possibly masked) that should be used as vector's lengths
        when plotting ``vectors`` array.
    :param cmap: (optional)
        Colormap to use for plotting colors.
        (default: ``gist_rainbow``)
    :param abs_levels: (optional)
        Iterable of absolute levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_abs_level: (optional)
        Values of minimal absolute level. Used with conjunction of ``k``
        argument for building sequence of absolute levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param rel_levels: (optional)
        Iterable of relative levels. If ``None`` then construct levels in other
        way. (default: ``None``)
    :param min_rel_level: (optional)
        Values of minimal relative level. Used with conjunction of ``k``
        argument for building sequence of relative levels. If ``None`` then
        construct levels in other way. (default: ``None``)
    :param k: (optional)
        Factor of incrementation for levels. (default: ``2.0``)
    :param colorbar_label: (optional)
        String to label colorbar. If ``None`` then don't label. (default:
        ``None``)
    :param plot_colorbar: (optional)
        If colors is set then should we plot colorbar? (default: ``True``).
    :param max_vector_value_length: (optional)
        Determines what part of the image is the length of the vector with
        maximum magnitude. E.g. if ``5`` then maximum value of vector quantity
        corresponds to arrow with length equal to 1/5 of the image length.
        (default: ``5``)
    :param mas_in_pixel: (optonal)
        Number of milliarcseconds in one pixel. If ``None`` then plot in pixels.
        (default: ``None``)
    :param vector_enlarge_factor: (optional)
        Additional factor to increase length of vectors representing direction and values of linear polarization.
    """
    matplotlib.rcParams['xtick.labelsize'] = label_size
    matplotlib.rcParams['ytick.labelsize'] = label_size
    matplotlib.rcParams['axes.titlesize'] = label_size
    matplotlib.rcParams['axes.labelsize'] = label_size
    matplotlib.rcParams['font.size'] = label_size
    matplotlib.rcParams['legend.fontsize'] = label_size
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    image = None
    if contours is not None:
        image = contours
    elif colors is not None and image is None:
        image = colors
    elif vectors is not None and image is None:
        image = vectors
    else:
        raise Exception("No image to plot")

    x = np.arange(image.shape[0]) - image.shape[0]/2
    y = np.arange(image.shape[1]) - image.shape[1]/2
    if mas_in_pixel is not None:
        x *= mas_in_pixel
        y *= mas_in_pixel

    # Optionally mask arrays
    if contours is not None and contours_mask is not None:
        contours = np.ma.array(contours, mask=contours_mask)
    if colors is not None and colors_mask is not None:
        colors = np.ma.array(colors, mask=colors_mask)
    if vectors is not None and vectors_mask is not None:
        vectors = np.ma.array(vectors, mask=vectors_mask)

    # Actually plotting
    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    else:
        ax = fig.get_axes()[0]

    if plot_title:
        title = ax.set_title(plot_title, fontsize='large')
    # Plot contours
    if contours is not None:
        if abs_levels is None:
            max_level = np.nanmax(contours)
            if rel_levels is not None:
                abs_levels = [-max_level] + [max_level * i for i in rel_levels]
            else:
                if min_abs_level is not None:
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                elif min_rel_level is not None:
                    min_abs_level = min_rel_level * max_level / 100.
                    n_max = int(math.ceil(math.log(max_level / min_abs_level, k)))
                else:
                    raise Exception("Not enough information for levels")
                abs_levels = [-min_abs_level] + [min_abs_level * k ** i for i in
                                                 range(n_max)]
        co = ax.contour(y, x, contours, abs_levels, colors=contour_color, linewidths=contour_linewidth)
    if colors is not None:
        im = ax.imshow(colors, interpolation='none',
                       origin='lower', extent=[y[0], y[-1], x[0], x[-1]],
                       cmap=plt.get_cmap(cmap), clim=color_clim)
    if vectors is not None:
        if vectors_values is not None:
            u = vectors_values * np.cos(vectors)
            v = vectors_values * np.sin(vectors)
            max_vector_value = np.max(np.abs(vectors_values))
            scale = max_vector_value_length*max_vector_value/vector_enlarge_factor
        else:
            u = np.cos(vectors)
            v = np.sin(vectors)
            scale = None

        if vectors_mask is not None:
            u = np.ma.array(u, mask=vectors_mask)
            v = np.ma.array(v, mask=vectors_mask)

        vec = ax.quiver(y[::vinc], x[::vinc], u[::vinc, ::vinc],
                        v[::vinc, ::vinc], angles='uv',
                        units='width', headwidth=0., headlength=0., scale=scale,
                        width=0.001, headaxislength=0., pivot='middle',
                        scale_units='width', color=vector_color, linewidths=quiver_linewidth, edgecolors='k')

    # Set equal aspect
    ax.set_aspect('auto')

    if colors is not None:
        if plot_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.00)
            cb = fig.colorbar(im, cax=cax)
            if colorbar_label is not None:
                cb.set_label(colorbar_label)

    # Saving output
    if outfile:
        if outdir is None:
            outdir = '.'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        path = os.path.join(outdir, outfile)
        plt.savefig("{}.png".format(path), bbox_inches='tight', dpi=300)

    if show:
        plt.ioff()
        plt.show()
    if close:
        plt.close()

    return fig


def beta(Gamma):
    """
    Velocity in units of speed of light [c].

    :param Gamma:
        Lorentz factor.
    """
    return np.sqrt(Gamma**2.-1.)/Gamma


def delta(Gamma, theta):
    """
    Doppler factor

    :param Gamma:
        Lorentz factor.
    :param theta:
        LOS angle [rad].
    """
    return 1./(Gamma*(1.-beta(Gamma)*np.cos(theta)))


def theta_plasma(theta_obs, Gamma):
    return np.arctan((np.sin(theta_obs)*np.sqrt(1 - beta(Gamma)**2))/(np.cos(theta_obs) - beta(Gamma)))


if __name__ == "__main__":

    CLI = argparse.ArgumentParser("Simulation of polarimetric VLBI observations of AGN jets with a spiral magnetic fields")
    CLI.add_argument("--template_uvfits",
                     type=str,
                     help="UVFITS file with the observed data used as a template for creating artificial data")
    CLI.add_argument("--redshift",
                     type=float,
                     default=0.5,
                     help="Redshift of the source (default: 0.5)")
    CLI.add_argument("--los_angle_deg",
                     type=float,
                     default=5.,
                     help="Viewing angle of the jet in degrees (default: 5)")
    CLI.add_argument("--cone_half_angle_deg",
                     type=float,
                     default=0.5,
                     help="Half-opening angle of the jet cone in degrees (default: 0.5)")
    CLI.add_argument("--Gamma",
                     type=float,
                     default=5.,
                     help="Lorentz-factor of the jet bulk motion (default: 5)")
    CLI.add_argument("--B_1_Gauss",
                     type=float,
                     default=0.5,
                     help="Magnetic field value (in G) at distance 1 pc from the BH (default: 0.5)")
    CLI.add_argument("--m",
                     type=float,
                     default=1.0,
                     help="Value of the exponent of the magnetic field power-law dependency on the distance along a jet: B ~ z^{-m} (default: 1.0)")
    CLI.add_argument("--pitch_angle_deg",
                     type=float,
                     default=60.,
                     help="Pitch angle (in the plasma frame!) of the jet spiral magnetic field in degrees (default: 60)")
    CLI.add_argument("--tangled_fraction",
                     type=float,
                     default=0.0,
                     help="Fraction of the tangled magnetic field (default: 0)")
    CLI.add_argument("--rot_angle_deg",
                     type=float,
                     default=0.,
                     help="Rotation (in degrees) of the model jet from South to positive RA axis (default: 0)")
    CLI.add_argument("--noise_scale_factor",
                     type=float,
                     default=1.0,
                     help="Coefficient to scale the noise. E.g. to decrease 10x times use ``0.1`` (default: 1)")

    args = CLI.parse_args()

    template_uvfits = args.template_uvfits


    rot_angle_deg = args.rot_angle_deg
    redshift = args.redshift
    los_angle_deg = args.los_angle_deg
    cone_half_angle_deg = args.cone_half_angle_deg
    Gamma = args.Gamma
    B_1 = args.B_1_Gauss
    m = args.m
    pitch_angle_deg = args.pitch_angle_deg
    tangled_fraction = args.tangled_fraction
    noise_scale_factor = args.noise_scale_factor

    if cone_half_angle_deg >= los_angle_deg:
        raise Exception("Don't look inside the cone! :)")

    print("LOS angle in the plasma frame = {:.1f} deg".format(np.rad2deg(theta_plasma(np.deg2rad(los_angle_deg), Gamma))))

    # Parameters of the model image with non-uniform pixel size
    n_along = 500
    n_across = 300
    lg_pixel_size_mas_min = np.log10(0.01)
    lg_pixel_size_mas_max = np.log10(0.1)

    cwd = os.getcwd()
    base_dir = cwd
    exec_dir = "{}/Release".format(base_dir)
    save_dir = os.path.join(base_dir, "results")
    # Create synthetic UVFITS files
    uvdata = UVData(template_uvfits)
    freq_ghz = np.round(uvdata.frequency/1e+09, 2)
    print("Using template UVFITS: {} at {} GHz".format(os.path.split(template_uvfits)[-1], freq_ghz))

    mapsize_clean = (1024, 0.1)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    ####################################################################################################################
    ############################# No need to change anything below this line ###########################################
    ####################################################################################################################

    stokes = ("I", "Q", "U")

    resolutions = np.logspace(lg_pixel_size_mas_min, lg_pixel_size_mas_max, n_along)
    print("Model jet extends up to {:.1f} mas!".format(np.sum(resolutions)))

    # Plot only jet emission and do not plot counter-jet?
    jet_only = True
    path_to_script = "{}/scripts/script_clean_rms".format(base_dir)

    images_i = list()
    images_q = list()
    images_u = list()
    images_pang = list()

    # Generate model image
    os.chdir(exec_dir)
    os.system(f"./simulate {freq_ghz} {redshift} {los_angle_deg} {cone_half_angle_deg} {Gamma} {B_1} {m}"
              f" {pitch_angle_deg} {tangled_fraction}")
    os.chdir(cwd)

    # Plot true image of polarization
    imagei = np.loadtxt(os.path.join(exec_dir, "{}/jet_image_{}_{}.txt".format(exec_dir, "i", freq_ghz)))
    imageq = np.loadtxt(os.path.join(exec_dir, "{}/jet_image_{}_{}.txt".format(exec_dir, "q", freq_ghz)))
    imageu = np.loadtxt(os.path.join(exec_dir, "{}/jet_image_{}_{}.txt".format(exec_dir, "u", freq_ghz)))
    mask = imagei == 0
    imagei[mask] = np.nan
    imageq[mask] = np.nan
    imageu[mask] = np.nan
    imagep = np.hypot(imageq, imageu)
    imagepang = 0.5*np.arctan2(imageu, imageq)

    min_abs_lev = 0.001*np.max(imagei)
    fig = plot_function(contours=imagei, colors=imagep, vectors=imagepang, vectors_values=None, min_rel_level=0.001,
                        vinc=10, contour_color="gray", vector_color="k", cmap="gist_rainbow", quiver_linewidth=0.01,
                        vector_enlarge_factor=8, colorbar_label="PPOL, Jy/pixel", contour_linewidth=1.0)
    fig = plot_function(contours=imagei, abs_levels=[0.01*np.max(imagei)], fig=fig, show=False, close=True)
    fig.savefig(os.path.join(save_dir, "true_pol.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Estimate thermal noise in real data
    noise = uvdata.noise(average_freq=False, use_V=False)
    # If one needs to decrease the noise added to the model visibilities - this is the way to do it
    for baseline, baseline_noise_std in noise.items():
        noise.update({baseline: noise_scale_factor*baseline_noise_std})

    stokes = ("I", "Q", "U", "V")
    jms = [JetImage(z=redshift, n_along=n_along, n_across=n_across,
                    lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
                    jet_side=True, rot=np.deg2rad(rot_angle_deg)) for _ in stokes]
    # cjms = [JetImage(z=z, n_along=n_along, n_across=n_across,
    #                  lg_pixel_size_mas_min=lg_pixel_size_mas_min, lg_pixel_size_mas_max=lg_pixel_size_mas_max,
    #                  jet_side=False) for _ in stokes]
    for i, stk in enumerate(stokes):
        jms[i].load_image_stokes(stk, "{}/jet_image_{}_{:.2f}.txt".format(exec_dir, stk.lower(), freq_ghz), scale=1.0)
        # cjms[i].load_image_stokes(stk, "../{}/cjet_image_{}_{}.txt".format(jetpol_run_directory, stk.lower(), freq_ghz), scale=1.0)

    # List of models (for J & CJ) for all stokes
    # js = [TwinJetImage(jms[i], cjms[i]) for i in range(len(stokes))]

    uvdata.zero_data()
    if jet_only:
        uvdata.substitute(jms)
    else:
        # uvdata.substitute(js)
        pass
    # Rotate EVPA also
    uvdata.rotate_evpa(np.deg2rad(rot_angle_deg))
    uvdata.noise_add(noise)
    # This should work for most of the MOJAVE data files
    downscale_by_freq = False
    artificial_uvfits = os.path.join(save_dir, "artificial.uvf")
    uvdata.save(artificial_uvfits, rewrite=True, downscale_by_freq=downscale_by_freq)

    # CLEAN synthetic UV-data
    stokes = ("I", "Q", "U")
    base_name = os.path.split(artificial_uvfits)[-1]
    base_name = base_name.split(".")[0]
    for stk in stokes:
        outfname = "{}_{}.fits".format(base_name, stk.lower())
        if os.path.exists(os.path.join(save_dir, outfname)):
            os.unlink(os.path.join(save_dir, outfname))
        clean_difmap(fname=artificial_uvfits, path=save_dir,
                     outfname=outfname, outpath=save_dir, stokes=stk.lower(),
                     mapsize_clean=mapsize_clean, path_to_script=path_to_script,
                     show_difmap_output=True)

    # Plot CLEAN images
    ccimages = dict()
    for stk in stokes:
        outfname = "{}_{}.fits".format(base_name, stk.lower())
        ccimages.update({stk: create_clean_image_from_fits_file(os.path.join(save_dir, outfname))})
    ipol = ccimages["I"].image
    beam = ccimages["I"].beam
    blc = None
    trc = None
    # Number of pixels in beam
    npixels_beam = int(np.pi*beam[0]*beam[1]/(4*np.log(2)*mapsize_clean[1]**2))
    plot_beam = (beam[0], beam[1], np.rad2deg(beam[2]))
    std = find_image_std(ipol, beam_npixels=npixels_beam, blc=blc, trc=trc)
    print("IPOL image std = {:.2f} mJy/beam".format(1000*std))
    if blc is None or trc is None:
        blc, trc = find_bbox(ipol, level=4*std, min_maxintensity_mjyperbeam=4*std,
                             min_area_pix=10*npixels_beam, delta=10)
        print("Find bbox results : ", blc, trc)
        if blc[0] == 0: blc = (blc[0]+1, blc[1])
        if blc[1] == 0: blc = (blc[0], blc[1]+1)
        if trc[0] == ipol.shape: trc = (trc[0]-1, trc[1])
        if trc[1] == ipol.shape: trc = (trc[0], trc[1]-1)
    masks_dict, ppol_quantile = pol_mask({stk: ccimages[stk].image for stk in stokes}, npixels_beam, n_sigma=4,
                                         return_quantile=True, blc=blc, trc=trc)
    ppol = np.hypot(ccimages["Q"].image, ccimages["U"].image)
    ppol = correct_ppol_bias(ipol, ppol, ccimages["Q"].image, ccimages["U"].image, npixels_beam)
    pang = 0.5*np.arctan2(ccimages["U"].image, ccimages["Q"].image)
    fpol = ppol/ipol

    images_i.append(ipol)
    images_q.append(ccimages["Q"].image)
    images_u.append(ccimages["U"].image)
    images_pang.append(np.ma.array(pang, mask=masks_dict["P"]))

    # Make a single epoch map
    # PPOL contours
    fig = iplot(ppol, x=ccimages["I"].x, y=ccimages["I"].y,
                min_abs_level=ppol_quantile, blc=blc, trc=trc,
                close=False, contour_color='black',
                plot_colorbar=False)
    # Add single IPOL contour and vectors of the PANG
    fig = iplot(contours=ipol, vectors=pang,
                x=ccimages["I"].x, y=ccimages["I"].y, vinc=4, contour_linewidth=0.25,
                vectors_mask=masks_dict["P"], abs_levels=[3*std], blc=blc, trc=trc,
                beam=plot_beam, close=True, show_beam=True, show=False,
                contour_color='gray', fig=fig, vector_color="black", plot_colorbar=False)
    axes = fig.get_axes()[0]
    axes.invert_xaxis()
    fig.savefig(os.path.join(save_dir, f"observed_pol.png"), dpi=600, bbox_inches="tight")
    plt.close()


    fig = iplot(ipol, fpol, x=ccimages["I"].x, y=ccimages["I"].y,
                min_abs_level=4*std, colors_mask=masks_dict["P"], color_clim=[0, 0.7], blc=blc, trc=trc,
                beam=plot_beam, close=True, colorbar_label="m", show_beam=True, show=False,
                cmap='gnuplot', contour_color='black', plot_colorbar=True,
                contour_linewidth=0.25)
    fig.savefig(os.path.join(save_dir, "observed_fpol.png"), dpi=600, bbox_inches="tight")
