import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, TextBox

def waveN(waveNm1, waveNm2, δ, ε, zNm2):
    return -waveNm1 + (2 * waveNm2) - (2 * δ**2 * (ε-np.fabs(zNm2)) * waveNm2)

def verifyε(steps, ε, wave0=1, dwave0=0):
    # E0 = ε, let hbar^2•a^2/m=1
    # zX = ε, z at which E0=a|x|
    steps = 2 ** steps
    δ = ε / steps
    z = np.arange(0, 4, δ)
    wave1 = wave0 - δ**2 * ε * wave0
    waveQ = np.zeros((len(z)))
    waveQ[0], waveQ[1] = wave0, wave1
    for i in range(2, len(z)):
        waveQ[i] = waveN(*waveQ[i-2:i], δ, ε, z[i-2]) 

    # calculate the "error" in the wave function 
    # i.e. how negative the tail is, or how much the tail is greater than its minimum value
    minVal = np.amin(waveQ)
    valid = 0 if (minVal - waveQ[-1] >= 0 and waveQ[-1] >= 0) else max(waveQ[-1]-minVal, -waveQ[-1])
    return z, waveQ, valid

εinit = 0.80857 # ±0.00001
εmin = 0.8085
εmax = 0.80865

stepinit = 13
stepmin = 12
stepmax = 15


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
    ax.set_ylim([-0.1, 1.1])

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.3, bottom=0.3)


    # Make a horizontal slider to control ε.
    axε = plt.axes([0.3, 0.175, 0.58, 0.03])
    ε_slider = Slider(
        ax=axε,
        label='ε',
        valmin=0,
        valmax=εmax,
        valinit=εinit,
    )

    # Make a vertically oriented slider to control δ
    axstep = plt.axes([0.175, 0.3, 0.0225, 0.58])
    step_slider = Slider(
        ax=axstep,
        label="Steps (for δ)",
        valmin=0,
        valmax=stepmax,
        valinit=stepinit,
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


    ax_step_min = plt.axes([0.05, 0.3, 0.1, 0.04])
    step_min = TextBox(ax_step_min, 'Step Min', initial=stepmin)
    l_smin = step_min.ax.get_children()[0]
    l_smin.set_y(1)
    l_smin.set_verticalalignment('bottom')
    l_smin.set_horizontalalignment('left')

    ax_step_max = plt.axes([0.05, 0.8, 0.1, 0.04])
    step_max = TextBox(ax_step_max, 'Step Max', initial=stepmax)
    l_smax = step_max.ax.get_children()[0]
    l_smax.set_y(1)
    l_smax.set_verticalalignment('bottom')
    l_smax.set_horizontalalignment('left')

    ax_ε_min = plt.axes([0.3, 0.1, 0.15, 0.04])
    ε_min = TextBox(ax_ε_min, 'ε Min', initial=εmin)

    ax_ε_max = plt.axes([0.75, 0.1, 0.15, 0.04])
    ε_max = TextBox(ax_ε_max, 'ε Max', initial=εmax)

    def update_slider(text, case):
        try:
            val = float(text)
            step = "s" == case[0]
            slider = step_slider if step else ε_slider
            if "min" in case and val < slider.valmax:
                slider.valmin = val
                if step:
                    slider.ax.set_ylim(val, None)
                else:
                    slider.ax.set_xlim(val, None)
                if val > slider.val:
                    slider.val=val
                    update(val)
            elif "max" in case and val > slider.valmin:
                slider.valmax = val
                if step:
                    slider.ax.set_ylim(None, val)
                else:
                    slider.ax.set_xlim(None, val)
                if val < slider.val:
                    slider.val=val
                    update(val)
            fig.canvas.draw_idle()
        except:
            pass

    step_min.on_submit(lambda text: update_slider(text, "smin"))
    step_max.on_submit(lambda text: update_slider(text, "smax"))
    ε_min.on_submit(lambda text: update_slider(text, "εmin"))
    ε_max.on_submit(lambda text: update_slider(text, "εmax"))
    update_slider(εmin, "εmin")
    update_slider(stepmin, "smin")

    
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        step_slider.reset()
        ε_slider.reset()
        step_min.set_val(stepmin)
        step_max.set_val(stepmax)
        ε_min.set_val(εmin)
        ε_max.set_val(εmax)
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
    steptol = np.linspace(12,15,tries)
    try: 
        errors = np.loadtxt(filename)
    except:
        errors = np.zeros((len(εtol), len(steptol)))#I was actually not dumb, wow what a miracle.
        for nε, ε in enumerate(εtol):
            print(nε, "/", len(εtol))
            for nstep, step in enumerate(steptol):
                err = verifyε(step, ε)[-1]
                errors[nε,nstep] = err
        np.savetxt(filename, errors)

    extent =[steptol[0],steptol[-1],εtol[0],εtol[-1]]
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(errors,origin='lower',interpolation='none', extent=extent, aspect="auto")
    fig.colorbar(im,shrink=0.8)
    ax.set_xlabel("Number of steps (log 2)")
    ax.set_ylabel("ε value")

    plt.show()
    
# Heatmap for the "error" in the wave function (calculation in line 22 above)
# with respect to a larger range of values of ε and δ (than above)
if False:
    tries = 100
    filename = "big"+str(tries)+str(tries)+".txt"
    εtol = np.linspace(0.5,0.9,tries)
    steptol = np.linspace(2,15,tries)
    try: 
        errors = np.loadtxt(filename)
    except:
        errors = np.zeros((len(εtol), len(steptol)))#I was actually not dumb, wow what a miracle.
        for nε, ε in enumerate(εtol):
            print(nε, "/", len(εtol))
            for nstep, step in enumerate(steptol):
                err = verifyε(step, ε)[-1]
                errors[nε,nstep] = err
        np.savetxt(filename, errors)

    extent =[steptol[0],steptol[-1],εtol[0],εtol[-1]]
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(errors,origin='lower',interpolation='none', extent=extent, aspect="auto", cmap=plt.get_cmap("plasma"))
    fig.colorbar(im,shrink=0.8)
    ax.set_xlabel("Number of steps (log 2)")
    ax.set_ylabel("ε value")
    
    rect=mpatches.Rectangle((12, εmin),3,εmax-εmin, 
                        fill=False,
                        color="white",
                       linewidth=1)
                       #facecolor="red")
    plt.gca().add_patch(rect)

    plt.show()
