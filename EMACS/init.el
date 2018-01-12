;; customize interface


;;;;; linum related ;;;;;
;; prog-mode-hook contains all the programming modes for emacs global setting for all of them 
(add-hook 'prog-mode-hook 'linum-mode)
(global-linum-mode 1);; fix not hooked julia in prog-mode-hook 
(setq column-number-mode t)
(setq linum-format "%d ")

;; Julia mode related
;; TODO: color-theme not seeem so good fix it 
(add-to-list 'load-path "/KUFS/scratch/okirnap/.emacs.d/julia-emacs");; you need to put mode directory
(require 'julia-mode)
