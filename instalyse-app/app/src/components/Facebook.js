// A Component to include a Facebook log in, only needed for further developements
//import React, { Component, useState } from 'react';
// import FacebookLogin from 'react-facebook-login';

// export default function Facebook() { //Class component Facebook Login
//     const [loggedIn, setLoggedIn] = useState(false)

//     const responseFacebook = (response) => {
//         //console.log("login result: ", response);
//         if (response.name) {
//             setLoggedIn(true);
//             //console.log("reaching if 1")
//         }
//         else {
//             setLoggedIn(false);
//             //console.log("wrong")
//         }
//     }

//     const componentClicked = () => {
//         //console.log("clicked");
//     }

//     let fbContent;
//     let fbTest;


//     if (loggedIn) {
//         fbContent = null;
//         //console.log("reaching if 2")
//         fbTest = "false";
//     }
//     else {
//         fbContent = (<FacebookLogin
//             appId="1381415035724085"
//             autoLoad={true}
//             fields=""
//             onClick={componentClicked}
//             callback={responseFacebook} />);
//             fbTest = "true";
//     }
//     return (
//         <div>
//             {fbContent}
//         </div>
//     )
// }