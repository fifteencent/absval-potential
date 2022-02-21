import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, TextBox

def waveN(waveNm1, waveNm2, δ, ε, zNm2):
    return -waveNm1 + (2 * waveNm2) - (2 * δ**2 * (ε-np.fabs(zNm2)) * waveNm2)

def verifyε(steps, ε, wave0=0.7005546694136285, complete=True):
    """Plot the wavefunction numerically using ε and steps.

    Arguments:
    steps       number of steps, log 2, taken from z=0 to z=ε (i.e. where E0=a|x|).
                    i.e. δ = ε * steps^(-2)
    ε           epsilon, the dimensionless parameter related to energy.
    wave0       ψ(0), determining the "amplitude" of the wave.
                    the given value normalizes the solution corresponding to the correct ε,
                    calculated with the help of Mathematica and Wikipedia
    complete    True: returns all values of ψ are desired regardless of correctness.
                False: Truncates ψ when error is detected. Calculates "valid" with remaining number of steps.
                    Plotting with complete=False is not as clear, but saves a lot of time.
    
    Return:
    z       An array of the z-values at which the wavefunction was evaluated. 
                A regular partition of 0 to 4 with step size δ (given above)
    waveQ   An array of the values of the wavefunction corresponding to the given ε,
                calculated numerically with the given δ.
    valid   A positive score denoting how "correct" ε is (given δ), with 0 being "correct"
    """
    # E0 = ε, let hbar^2•a^2/m = 1
    # E0 = a|x| when z = ε

    steps = 2 ** steps
    δ = ε / steps
    z = np.arange(0, 4, δ)
    wave1 = wave0 - δ**2 * ε * wave0
    waveQ = np.zeros((len(z)))
    waveQ[0], waveQ[1] = wave0, wave1
    valid = 0
    for i in range(2, len(z)):
        waveQ[i] = waveN(*waveQ[i-2:i], δ, ε, z[i-2]) 
        valid = 0 if (waveQ[i-2] - waveQ[i] >= 0 and waveQ[i] >= 0) \
                else (1 if not complete else max(waveQ[i]-np.amin(waveQ), -waveQ[i]))
        if not complete and valid != 0:
            valid = len(z)- i
            break

    # calculate the "error" in the wave function 
    # i.e. how negative the tail is, or how much the tail is greater than its minimum value
    return z, waveQ, valid

εinit = 0.80857 # actual is ±0.00001
εmin = 0.8085
εmax = 0.80865

stepinit = 13
stepmin = 12
stepmax = 15

def getErrors(filename, εtol, steptol, **argc):
    try: 
        errors = np.loadtxt(filename)
    except:
        errors = np.zeros((len(εtol), len(steptol)))#I was actually not dumb, wow what a miracle.
        for nε, ε in enumerate(εtol):
            print(nε, "/", len(εtol))
            for nstep, step in enumerate(steptol):
                err = verifyε(step, ε, **argc)[-1]
                errors[nε,nstep] = err
        np.savetxt(filename, errors)
    return errors

# Plots wavefunction for corresponding ε and δ values (controlled with sliders).
# Red and green correspond to invalid and valid solutions respectively.
if True:
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    oldx, oldy, valid = verifyε(stepinit, εinit)
    color = "green" if valid == 0 else "red"
    line, = plt.plot(oldx, oldy, lw=2, color=color)
    ax.set_xlabel('z')
    ax.set_ylabel('ψ')
    plt.title("ψ(x)")
    ax.set_ylim([-0.05, 0.8])
    plt.xticks(np.arange(0, 4.1, 1))
    plt.yticks(np.arange(0, 0.81, 0.2))

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.3, bottom=0.3)


    # Make a horizontal slider to control ε (min gets adjusted right before rendering)
    axstep = plt.axes([0.3, 0.175, 0.58, 0.03])
    step_slider = Slider(
        ax=axstep,
        label="Steps (log2, for δ)",
        valmin=0,
        valmax=stepmax,
        valinit=stepinit,
    )

    # Make a vertically oriented slider to control δ (min gets adjusted right before rendering)
    axε = plt.axes([0.175, 0.3, 0.0225, 0.58])
    ε_slider = Slider(
        ax=axε,
        label='ε',
        valmin=0,
        valmax=εmax,
        valinit=εinit,
        orientation="vertical"
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        newx, newy, valid = verifyε(step_slider.val, ε_slider.val)
        line.set_xdata(newx)
        line.set_ydata(newy)
        line.set_color("green" if valid == 0 else "red")
        fig.canvas.draw_idle()

    # register the update function with each slider
    step_slider.on_changed(update)
    ε_slider.on_changed(update)


    ax_ε_min = plt.axes([0.05, 0.3, 0.1, 0.04])
    εmin_box = TextBox(ax_ε_min, 'ε Min', initial=εmin)
    
    # reposition label
    l_εmin = εmin_box.ax.get_children()[0]
    l_εmin.set_y(1)
    l_εmin.set_verticalalignment('bottom')
    l_εmin.set_horizontalalignment('left')

    ax_ε_max = plt.axes([0.05, 0.8, 0.1, 0.04])
    εmax_box = TextBox(ax_ε_max, 'ε Max', initial=εmax)
    
    # reposition label
    l_εmax = εmax_box.ax.get_children()[0]
    l_εmax.set_y(1)
    l_εmax.set_verticalalignment('bottom')
    l_εmax.set_horizontalalignment('left')

    ax_step_min = plt.axes([0.3, 0.1, 0.15, 0.04])
    smin_box = TextBox(ax_step_min, 'Step Min', initial=stepmin)

    ax_step_max = plt.axes([0.75, 0.1, 0.15, 0.04])
    smax_box = TextBox(ax_step_max, 'Step Max', initial=stepmax)

    def update_slider(text, case):
        step = "s" == case[0]
        try:
            val = float(text)
            slider = step_slider if step else ε_slider
            if "min" in case and 0 < val < slider.valmax:
                slider.valmin = val
                if step:
                    slider.ax.set_xlim(val, None)
                else:
                    slider.ax.set_ylim(val, None)
                if val > slider.val:
                    slider.val=val
                    update(val)
            elif "max" in case and val > slider.valmin:
                slider.valmax = val
                if step:
                    slider.ax.set_xlim(None, val)
                else:
                    slider.ax.set_ylim(None, val)
                if val < slider.val:
                    slider.val=val
                    update(val)
            fig.canvas.draw_idle()
        except:
            textbox = smin_box if step else εmin_box
            if "min" in case:
                textbox.set_val(slider.valmin)
            else:
                textbox.set_val(slider.valmax)
            pass

    smin_box.on_submit(lambda text: update_slider(text, "smin"))
    smax_box.on_submit(lambda text: update_slider(text, "smax"))
    εmin_box.on_submit(lambda text: update_slider(text, "εmin"))
    εmax_box.on_submit(lambda text: update_slider(text, "εmax"))
    update_slider(εmin, "εmin")
    update_slider(stepmin, "smin")

    
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        step_slider.reset()
        ε_slider.reset()
        smin_box.set_val(stepmin)
        smax_box.set_val(stepmax)
        εmin_box.set_val(εmin)
        εmax_box.set_val(εmax)
        update_slider(εmin, "εmin")
        update_slider(stepmin, "smin")
        update_slider(εmax, "εmax")
        update_slider(stepmax, "smax")

    button.on_clicked(reset)

    plt.show()

# Heatmap for the "error" in the wave function (calculation in line 22 above)
# with respect to various values of ε and δ (close to the actual value of ε)
if False:
    tries = 10
    filename = str(tries)+str(tries)+".txt"
    εtol = np.linspace(εmin,εmax,tries)
    steptol = np.linspace(stepmin,stepmax,tries)
    errors = getErrors(filename, εtol, steptol)

    extent =[steptol[0],steptol[-1],εtol[0],εtol[-1]]
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(errors, \
        origin='lower',interpolation='none', extent=extent, aspect="auto")
    fig.colorbar(im,shrink=0.8)
    ax.set_xlabel("Number of steps (log 2)")
    ax.set_ylabel("ε value")

    plt.show()
    
# Heatmap for the "error" in the wave function (calculation in line 22 above)
# with respect to a larger range of values of ε and δ (than above)
if False:
    tries = 100
    complete=True
    filename = "big"+("" if complete else "n")+str(tries)+str(tries)+".txt"
    εtol = np.linspace(0.5,0.9,tries)
    steptol = np.linspace(2,15,tries)
    errors = getErrors(filename, εtol, steptol)

    extent =[steptol[0],steptol[-1],εtol[0],εtol[-1]]
    fig, ax = plt.subplots(figsize=(8,8))

    im = ax.imshow(errors, \
        origin='lower',interpolation='none', extent=extent, aspect="auto", cmap=plt.get_cmap("plasma"), norm=colors.PowerNorm(0.6))
    fig.colorbar(im,shrink=0.8)
    ax.set_xlabel("Number of steps (log 2)")
    ax.set_ylabel("ε value")
    
    # to show region in more zoomed in plot (2nd if statement)
    rect=mpatches.Rectangle((12, εmin),3,εmax-εmin, 
                        fill=False,
                        color="white",
                       linewidth=1)
                       #facecolor="red")
    plt.gca().add_patch(rect)

    plt.show()
