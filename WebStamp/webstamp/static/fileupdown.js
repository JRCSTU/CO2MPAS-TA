/*global saveAs, self*/

"use strict";
$(document).ready(function(event){
    var upload_filename = $("#upload-filename");
    var upload_spinner = $("#upload-spinner");
    var dice_report = $("#dice_report");
    var download_machine = $("#download-machine");
    var download_filename = $("#download-filename");
    var download_btn = $("#download-btn");

    if (!dice_report[0].readOnly) {
        // Before stamping, hide download-stamp machinery.
        //
        download_machine.addClass('hidden');

        // Upload-button populates dice-report Text-Area.
        //
        upload_filename.change(function(event) {
            //var fpath = upload_filename[0].files[0].name;
            var file =  event.target.files[0];
            if (file) {
                upload_spinner.removeClass("hidden");
                var reader = new FileReader();
                reader.onload = function(event) {
                    dice_report[0].value = event.target.result;
                    upload_spinner.addClass("hidden");
                };

                reader.readAsText(file);
            }
        });
    } else {
        download_machine.removeClass('hidden');

        // Download-button saves locally the contents of dice-report Text-Area.
        //
        download_btn.click(function(event){
            saveAs(
                new self.Blob(
                    [dice_report.value],
                    {type: "text/plain;charset=" + document.characterSet}
                ),
                (download_filename.value || download_filename[0].placeholder || "stamp.txt")
            );
        });
    }
});
